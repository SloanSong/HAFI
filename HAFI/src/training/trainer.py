import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from lifelines.utils import concordance_index
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .loss import RankingLoss


class AdvancedTrainer:
    def __init__(self, model, train_loader, test_loader, scaler, device, epochs):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.device = device
        self.epochs = epochs
        self.optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=5e-5)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=2, eta_min=1e-6)
        self.criterion = nn.MSELoss()
        self.ranking_criterion = RankingLoss(margin=0.8)
        self.best_ci = 0
        self.history = {'train_loss': [], 'test_loss': [], 'mse': [], 'rmse': [], 'rm2': [], 'ci': []}

    def compute_metrics(self, y_true, y_pred):
        y_true = self.scaler.inverse_transform(y_true.reshape(-1,1)).flatten()
        y_pred = self.scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
        r2 = r2_score(y_true, y_pred)
        n = len(y_true)
        rm2 = 1 - (1 - r2) * (n - 1) / (n - 2)
        return {'mse': mean_squared_error(y_true, y_pred),
                'rmse': math.sqrt(mean_squared_error(y_true, y_pred)),
                'rm2': rm2,
                'ci': concordance_index(y_true, y_pred)}

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        for drug, protein, d_seq, p_seq, aff, fp, prot_feat in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
            if drug.num_graphs == 0: continue
            drug = drug.to(self.device)
            protein = protein.to(self.device)
            d_seq = d_seq.to(self.device)
            p_seq = p_seq.to(self.device)
            aff = aff.to(self.device).unsqueeze(1)
            fp = fp.to(self.device)
            prot_feat = prot_feat.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(drug, protein, d_seq, p_seq, fp, prot_feat)

            _, _, fp_kl, fp_recon = self.model.drug_fp_vae(fp, compute_loss=True)
            _, _, prot_kl, prot_recon = self.model.protein_phys_vae(prot_feat, compute_loss=True)
            vae_loss = fp_kl + fp_recon + prot_kl + prot_recon

            mse_loss = self.criterion(pred, aff)
            rank_loss = self.ranking_criterion(pred, aff)
            loss = mse_loss + 0.5 * rank_loss + 0.1 * vae_loss

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item() * len(aff)
        return total_loss / len(self.train_loader.dataset)

    def evaluate(self):
        self.model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for drug, protein, d_seq, p_seq, aff, fp, prot_feat in self.test_loader:
                if drug.num_graphs == 0: continue
                drug = drug.to(self.device)
                protein = protein.to(self.device)
                d_seq = d_seq.to(self.device)
                p_seq = p_seq.to(self.device)
                aff = aff.to(self.device).unsqueeze(1)
                fp = fp.to(self.device)
                prot_feat = prot_feat.to(self.device)
                pred = self.model(drug, protein, d_seq, p_seq, fp, prot_feat)
                all_pred.append(pred.cpu())
                all_true.append(aff.cpu())
        all_pred = torch.cat(all_pred).numpy()
        all_true = torch.cat(all_true).numpy()
        return self.compute_metrics(all_true, all_pred)

    def train(self):
        print(f"Training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            metrics = self.evaluate()
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(metrics['mse'])
            self.history['mse'].append(metrics['mse'])
            self.history['rmse'].append(metrics['rmse'])
            self.history['rm2'].append(metrics['rm2'])
            self.history['ci'].append(metrics['ci'])
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, MSE={metrics['mse']:.4f}, CI={metrics['ci']:.4f}, Rm²={metrics['rm2']:.4f}")
            if metrics['ci'] > self.best_ci:
                self.best_ci = metrics['ci']
                torch.save(self.model.state_dict(), 'best_model_hafidta.pth')
                print(f"  New best CI={self.best_ci:.4f} - saved.")
        self.plot_training_history()

    def plot_training_history(self):
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.plot(self.history['train_loss'], label='Train Loss'); plt.plot(self.history['test_loss'], label='Test Loss'); plt.legend(); plt.grid()
        plt.subplot(1,3,2); plt.plot(self.history['ci'], label='CI', color='green'); plt.plot(self.history['rm2'], label='Rm²', color='blue'); plt.legend(); plt.grid()
        plt.subplot(1,3,3); plt.plot(self.history['rmse'], label='RMSE', color='red'); plt.legend(); plt.grid()
        plt.tight_layout(); plt.savefig('training_history_hafidta.png', dpi=300); print("Plot saved.")


class KFoldTrainer:
    def __init__(self, model_class, model_args, kfold_loaders, scaler, device, epochs):
        self.model_class = model_class
        self.model_args = model_args
        self.kfold_loaders = kfold_loaders
        self.scaler = scaler
        self.device = device
        self.epochs = epochs
        self.fold_results = []
        self.all_histories = []

    def _create_model(self):
        model = self.model_class(**self.model_args)
        return model.to(self.device)

    def train_fold(self, fold_idx, train_loader, test_loader):
        model = self._create_model()
        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=5e-5)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-7)
        criterion = nn.MSELoss()
        ranking_criterion = RankingLoss(margin=0.8)
        best_ci = 0
        history = {'train_loss': [], 'test_loss': [], 'mse': [], 'rmse': [], 'rm2': [], 'ci': []}

        print(f"\n=== Fold {fold_idx+1}/{len(self.kfold_loaders)} ===")
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            for drug, protein, d_seq, p_seq, aff, fp, prot_feat in tqdm(train_loader, desc=f"Fold {fold_idx+1} Epoch {epoch+1}"):
                if drug.num_graphs == 0: continue
                drug = drug.to(self.device)
                protein = protein.to(self.device)
                d_seq = d_seq.to(self.device)
                p_seq = p_seq.to(self.device)
                aff = aff.to(self.device).unsqueeze(1)
                fp = fp.to(self.device)
                prot_feat = prot_feat.to(self.device)

                optimizer.zero_grad()

                pred = model(drug, protein, d_seq, p_seq, fp, prot_feat)

                _, _, fp_kl, fp_recon = model.drug_fp_vae(fp, compute_loss=True)
                _, _, prot_kl, prot_recon = model.protein_phys_vae(prot_feat, compute_loss=True)
                vae_loss = fp_kl + fp_recon + prot_kl + prot_recon

                mse_loss = criterion(pred, aff)
                rank_loss = ranking_criterion(pred, aff)
                loss = mse_loss + 0.5 * rank_loss + 0.1 * vae_loss

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item() * len(aff)

            train_loss = total_loss / len(train_loader.dataset)

            model.eval()
            all_pred, all_true = [], []
            with torch.no_grad():
                for drug, protein, d_seq, p_seq, aff, fp, prot_feat in test_loader:
                    if drug.num_graphs == 0: continue
                    drug = drug.to(self.device)
                    protein = protein.to(self.device)
                    d_seq = d_seq.to(self.device)
                    p_seq = p_seq.to(self.device)
                    aff = aff.to(self.device).unsqueeze(1)
                    fp = fp.to(self.device)
                    prot_feat = prot_feat.to(self.device)
                    pred = model(drug, protein, d_seq, p_seq, fp, prot_feat)
                    all_pred.append(pred.cpu())
                    all_true.append(aff.cpu())
            all_pred = torch.cat(all_pred).numpy()
            all_true = torch.cat(all_true).numpy()

            y_true = self.scaler.inverse_transform(all_true.reshape(-1,1)).flatten()
            y_pred = self.scaler.inverse_transform(all_pred.reshape(-1,1)).flatten()
            r2 = r2_score(y_true, y_pred)
            n = len(y_true)
            rm2 = 1 - (1 - r2) * (n - 1) / (n - 2)
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': math.sqrt(mean_squared_error(y_true, y_pred)),
                'rm2': rm2,
                'ci': concordance_index(y_true, y_pred)
            }

            history['train_loss'].append(train_loss)
            history['test_loss'].append(metrics['mse'])
            history['mse'].append(metrics['mse'])
            history['rmse'].append(metrics['rmse'])
            history['rm2'].append(metrics['rm2'])
            history['ci'].append(metrics['ci'])

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, MSE={metrics['mse']:.4f}, CI={metrics['ci']:.4f}, Rm²={metrics['rm2']:.4f}")

            if metrics['ci'] > best_ci:
                best_ci = metrics['ci']
                torch.save(model.state_dict(), f'best_model_hafidta_fold{fold_idx+1}.pth')

        self.fold_results.append(best_ci)
        self.all_histories.append(history)
        print(f"Fold {fold_idx+1} completed. Best CI: {best_ci:.4f}")
        return best_ci

    def train(self):
        print(f"Starting {len(self.kfold_loaders)}-fold cross validation...")
        for fold_idx, train_loader, test_loader in self.kfold_loaders:
            self.train_fold(fold_idx, train_loader, test_loader)

        self._print_summary()

    def _print_summary(self):
        print("\n" + "="*50)
        print("KFOLD CROSS VALIDATION SUMMARY")
        print("="*50)
        for i, ci in enumerate(self.fold_results):
            print(f"Fold {i+1}: CI = {ci:.4f}")
        print("-"*50)
        print(f"Mean CI: {np.mean(self.fold_results):.4f} ± {np.std(self.fold_results):.4f}")
        print(f"Best CI: {max(self.fold_results):.4f}")
        print(f"Worst CI: {min(self.fold_results):.4f}")

    def plot_kfold_history(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        for i, history in enumerate(self.all_histories):
            plt.plot(history['train_loss'], label=f'Fold {i+1} Train')
            plt.plot(history['test_loss'], label=f'Fold {i+1} Test')
        plt.title('Loss')
        plt.legend()
        plt.grid()

        plt.subplot(1, 3, 2)
        for i, history in enumerate(self.all_histories):
            plt.plot(history['ci'], label=f'Fold {i+1}')
        plt.title('CI')
        plt.legend()
        plt.grid()

        plt.subplot(1, 3, 3)
        for i, history in enumerate(self.all_histories):
            plt.plot(history['rm2'], label=f'Fold {i+1}')
        plt.title('Rm²')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig('kfold_training_history.png', dpi=300)
        print("KFold training history plot saved.")


class NestedCVTrainer:
    def __init__(self, model_class, model_args, nested_cv_loaders, scaler, device, epochs, seed=42):
        self.model_class = model_class
        self.model_args = model_args
        self.nested_cv_loaders = nested_cv_loaders
        self.scaler = scaler
        self.device = device
        self.epochs = epochs
        self.seed = seed
        self.all_results = []
        self.all_histories = []

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _create_model(self, seed=None):
        if seed is not None:
            self._set_seed(seed)
        model = self.model_class(**self.model_args)
        return model.to(self.device)

    def _compute_metrics(self, y_true, y_pred):
        y_true = self.scaler.inverse_transform(y_true.reshape(-1,1)).flatten()
        y_pred = self.scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
        r2 = r2_score(y_true, y_pred)
        n = len(y_true)
        rm2 = 1 - (1 - r2) * (n - 1) / (n - 2)
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': math.sqrt(mean_squared_error(y_true, y_pred)),
            'rm2': rm2,
            'ci': concordance_index(y_true, y_pred)
        }

    def _evaluate_model(self, model, loader):
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for drug, protein, d_seq, p_seq, aff, fp, prot_feat in loader:
                if drug.num_graphs == 0: continue
                drug = drug.to(self.device)
                protein = protein.to(self.device)
                d_seq = d_seq.to(self.device)
                p_seq = p_seq.to(self.device)
                aff = aff.to(self.device).unsqueeze(1)
                fp = fp.to(self.device)
                prot_feat = prot_feat.to(self.device)
                pred = model(drug, protein, d_seq, p_seq, fp, prot_feat)
                all_pred.append(pred.cpu())
                all_true.append(aff.cpu())
        all_pred = torch.cat(all_pred).numpy()
        all_true = torch.cat(all_true).numpy()
        return self._compute_metrics(all_true, all_pred)

    def train_inner_fold(self, model, train_loader, val_loader, optimizer, scheduler, criterion, ranking_criterion):
        best_val_ci = 0
        best_state_dict = None

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            for drug, protein, d_seq, p_seq, aff, fp, prot_feat in train_loader:
                if drug.num_graphs == 0: continue
                drug = drug.to(self.device)
                protein = protein.to(self.device)
                d_seq = d_seq.to(self.device)
                p_seq = p_seq.to(self.device)
                aff = aff.to(self.device).unsqueeze(1)
                fp = fp.to(self.device)
                prot_feat = prot_feat.to(self.device)

                optimizer.zero_grad()

                pred = model(drug, protein, d_seq, p_seq, fp, prot_feat)

                _, _, fp_kl, fp_recon = model.drug_fp_vae(fp, compute_loss=True)
                _, _, prot_kl, prot_recon = model.protein_phys_vae(prot_feat, compute_loss=True)
                vae_loss = fp_kl + fp_recon + prot_kl + prot_recon

                mse_loss = criterion(pred, aff)
                rank_loss = ranking_criterion(pred, aff)
                loss = mse_loss + 0.5 * rank_loss + 0.1 * vae_loss

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * len(aff)
            scheduler.step()

            val_metrics = self._evaluate_model(model, val_loader)
            if val_metrics['ci'] > best_val_ci:
                best_val_ci = val_metrics['ci']
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        return best_val_ci, best_state_dict

    def train_outer_fold(self, repeat, outer_fold, test_loader, inner_loaders):
        inner_best_cis = []
        all_inner_state_dicts = []

        print(f"\n{'='*60}")
        print(f"Repeat {repeat+1}, Outer Fold {outer_fold+1}")
        print(f"{'='*60}")

        for inner_fold_idx, train_loader, val_loader in inner_loaders:
            print(f"\n--- Inner Fold {inner_fold_idx+1}/{len(inner_loaders)} ---")
            inner_seed = self.seed + repeat * 1000 + outer_fold * 100 + inner_fold_idx
            model = self._create_model(seed=inner_seed)
            optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=5e-5)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-7)
            criterion = nn.MSELoss()
            ranking_criterion = RankingLoss(margin=0.8)

            best_val_ci, best_state_dict = self.train_inner_fold(
                model, train_loader, val_loader, optimizer, scheduler, criterion, ranking_criterion
            )

            inner_best_cis.append(best_val_ci)
            all_inner_state_dicts.append(best_state_dict)
            print(f"  Inner Fold {inner_fold_idx+1} Best Val CI: {best_val_ci:.4f} (seed={inner_seed})")

        best_inner_idx = np.argmax(inner_best_cis)
        best_inner_ci = inner_best_cis[best_inner_idx]
        print(f"\nBest Inner Fold: {best_inner_idx+1} with CI: {best_inner_ci:.4f}")

        final_model = self._create_model(seed=self.seed + repeat * 1000 + outer_fold * 100 + best_inner_idx)
        final_model.load_state_dict(all_inner_state_dicts[best_inner_idx])

        print("\nEvaluating on independent test set...")
        test_metrics = self._evaluate_model(final_model, test_loader)
        print(f"Test Set Results: MSE={test_metrics['mse']:.4f}, RMSE={test_metrics['rmse']:.4f}, CI={test_metrics['ci']:.4f}, Rm²={test_metrics['rm2']:.4f}")

        torch.save(final_model.state_dict(), f'best_model_repeat{repeat+1}_outer{outer_fold+1}.pth')

        return {
            'repeat': repeat,
            'outer_fold': outer_fold,
            'inner_cis': inner_best_cis,
            'best_inner_ci': best_inner_ci,
            'test_metrics': test_metrics
        }

    def train(self):
        print(f"Starting Nested Cross Validation: {len(self.nested_cv_loaders)} outer folds across {len(set(r['repeat'] for r in self.nested_cv_loaders))} repeats")

        for cv_item in self.nested_cv_loaders:
            result = self.train_outer_fold(
                cv_item['repeat'],
                cv_item['outer_fold'],
                cv_item['test_loader'],
                cv_item['inner_loaders']
            )
            self.all_results.append(result)

        self._print_summary()

    def _print_summary(self):
        print("\n" + "="*60)
        print("NESTED CROSS VALIDATION FINAL SUMMARY")
        print("="*60)

        repeats = sorted(set(r['repeat'] for r in self.all_results))
        for repeat in repeats:
            repeat_results = [r for r in self.all_results if r['repeat'] == repeat]
            test_cis = [r['test_metrics']['ci'] for r in repeat_results]
            print(f"\nRepeat {repeat+1}:")
            for i, r in enumerate(repeat_results):
                print(f"  Outer Fold {i+1}: CI = {r['test_metrics']['ci']:.4f}")
            print(f"  Mean CI: {np.mean(test_cis):.4f} ± {np.std(test_cis):.4f}")

        all_test_cis = [r['test_metrics']['ci'] for r in self.all_results]
        all_test_mses = [r['test_metrics']['mse'] for r in self.all_results]
        all_test_rmses = [r['test_metrics']['rmse'] for r in self.all_results]
        all_test_rm2s = [r['test_metrics']['rm2'] for r in self.all_results]

        print("\n" + "-"*60)
        print("OVERALL RESULTS (All Repeats Combined)")
        print("-"*60)
        print(f"Total number of outer folds: {len(self.all_results)}")
        print(f"Mean CI: {np.mean(all_test_cis):.4f} ± {np.std(all_test_cis):.4f}")
        print(f"Mean MSE: {np.mean(all_test_mses):.4f} ± {np.std(all_test_mses):.4f}")
        print(f"Mean RMSE: {np.mean(all_test_rmses):.4f} ± {np.std(all_test_rmses):.4f}")
        print(f"Mean Rm²: {np.mean(all_test_rm2s):.4f} ± {np.std(all_test_rm2s):.4f}")
        print(f"Best CI: {max(all_test_cis):.4f}")
        print(f"Worst CI: {min(all_test_cis):.4f}")