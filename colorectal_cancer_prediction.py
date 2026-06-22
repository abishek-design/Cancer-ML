import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import warnings
import logging
from typing import Tuple, List, Dict, Any

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import shap

# Configure logging and warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# Set Matplotlib style for professional outputs
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['figure.dpi'] = 100

class ColorectalCancerProject:
    def __init__(self, data_path: str, target_column: str = 'target'):
        """
        Initialize the project pipeline.
        :param data_path: Path to the dataset (CSV format).
        :param target_column: The name of the target variable to predict.
        """
        self.data_path = data_path
        self.target_column = target_column
        self.df = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.models = {}
        self.best_model_name = None
        self.best_model = None
        self.best_pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.evaluation_results = {}
        
        # Ensure output directory for charts exists
        os.makedirs('outputs', exist_ok=True)

    # ==========================================
    # Phase 1: Data Loading and Inspection
    # ==========================================
    def load_and_inspect_data(self):
        logging.info("--- Phase 1: Data Loading and Inspection ---")
        try:
            self.df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            logging.warning(f"File {self.data_path} not found. Generating a synthetic dataset for demonstration purposes.")
            self._generate_synthetic_data()
        
        # Drop Participant ID
        if 'Participant_ID' in self.df.columns:
            self.df.drop(columns=['Participant_ID'], inplace=True)
            
        # Dynamically identify column types
        if self.target_column not in self.df.columns:
            # Guess target column if not found
            possible_targets = ['crc_risk', 'cancer', 'risk', 'diagnosis', 'target', 'y']
            found = False
            for pt in possible_targets:
                for col in self.df.columns:
                    if pt.lower() in col.lower():
                        if pt == 'y' and len(col) > 1 and col.lower() != 'y':
                            continue
                        self.target_column = col
                        found = True
                        break
                if found:
                    break
        
        if self.target_column not in self.df.columns:
            self.target_column = self.df.columns[-1] # fallback to last column
            
        logging.info(f"Target column identified as: {self.target_column}")

        print("\n--- Dataset Overview ---")
        print(f"Shape: {self.df.shape}")
        print("\nData Types:")
        print(self.df.dtypes)
        print("\nFirst 5 Rows:")
        display_df = self.df.head(5) if len(self.df) > 5 else self.df
        print(display_df)
        print("\nMissing Values Summary:")
        print(self.df.isnull().sum()[self.df.isnull().sum() > 0])
        print(f"\nDuplicate Records: {self.df.duplicated().sum()}")
        print("\nDescriptive Statistics:")
        print(self.df.describe())

    def _generate_synthetic_data(self):
        """Generates synthetic dataset if the real one is missing for demonstration."""
        np.random.seed(42)
        n_samples = 1000
        self.df = pd.DataFrame({
            'Age': np.random.normal(55, 12, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'BMI': np.random.normal(28, 5, n_samples),
            'Smoking_Status': np.random.choice(['Never', 'Former', 'Current'], n_samples),
            'Alcohol_Consumption': np.random.choice(['Low', 'Moderate', 'High'], n_samples),
            'Physical_Activity': np.random.choice(['Low', 'Moderate', 'High'], n_samples),
            'Sleep_Hours': np.random.normal(7, 1.5, n_samples),
            'Red_Meat_Consumption': np.random.normal(3, 1.5, n_samples), # servings/week
            'Processed_Food_Consumption': np.random.normal(4, 2, n_samples),
            'Fiber_Intake': np.random.normal(20, 8, n_samples), # grams/day
            'Fruit_Consumption': np.random.normal(2, 1, n_samples), # servings/day
            'Vegetable_Consumption': np.random.normal(3, 1.5, n_samples),
            'Water_Intake': np.random.normal(2, 0.8, n_samples) # Liters
        })
        
        # Create a synthetic relationship for the target
        risk_score = (
            self.df['Age'] * 0.05 + 
            (self.df['BMI'] > 30).astype(int) * 1.5 +
            (self.df['Smoking_Status'] == 'Current').astype(int) * 2.0 +
            (self.df['Red_Meat_Consumption'] * 0.3) -
            (self.df['Fiber_Intake'] * 0.1) -
            (self.df['Physical_Activity'] == 'High').astype(int) * 1.5
        )
        prob = 1 / (1 + np.exp(-risk_score + np.median(risk_score)))
        self.df['Colorectal_Cancer_Risk'] = (prob > 0.5).astype(int)
        self.target_column = 'Colorectal_Cancer_Risk'

    # ==========================================
    # Phase 2: Data Cleaning
    # ==========================================
    def clean_data(self):
        logging.info("--- Phase 2: Data Cleaning ---")
        
        # 1. Remove duplicates
        initial_shape = self.df.shape
        self.df.drop_duplicates(inplace=True)
        logging.info(f"Removed {initial_shape[0] - self.df.shape[0]} duplicate rows.")

        # 2. Identify dynamic column lists
        self.categorical_cols = [col for col in self.df.columns if self.df[col].dtype == 'object' and col != self.target_column]
        self.numeric_cols = [col for col in self.df.columns if self.df[col].dtype in ['int64', 'float64'] and col != self.target_column]

        # 3. Missing Value Treatment
        # Numeric: Fill with median, Categorical: Fill with mode
        for col in self.numeric_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
                
        for col in self.categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        # 4. Outlier Handling (Cap extreme values at 1st and 99th percentiles for numerics)
        for col in self.numeric_cols:
            lower = self.df[col].quantile(0.01)
            upper = self.df[col].quantile(0.99)
            self.df[col] = np.clip(self.df[col], lower, upper)
            
        logging.info("Data cleaning complete. Missing values imputed and outliers clipped.")

    # ==========================================
    # Phase 3: Exploratory Data Analysis (EDA)
    # ==========================================
    def perform_eda(self):
        logging.info("--- Phase 3: Exploratory Data Analysis (EDA) ---")
        
        self._plot_target_distribution()
        self._plot_correlation_heatmap()
        self._plot_demographics()
        self._plot_lifestyle_factors()
        self._plot_dietary_factors()
        self._plot_comparative_analysis()

    def _plot_target_distribution(self):
        plt.figure(figsize=(8, 6))
        counts = self.df[self.target_column].value_counts()
        bars = plt.bar(counts.index.astype(str), counts.values, color=['#3498db', '#e74c3c'])
        plt.title('Target Class Distribution', fontweight='bold')
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('outputs/01_target_distribution.png')
        plt.close()

    def _plot_correlation_heatmap(self):
        if not self.numeric_cols: return
        plt.figure(figsize=(12, 10))
        corr = self.df[self.numeric_cols].corr()
        
        # Simple matplotlib heatmap
        cax = plt.matshow(corr, cmap='coolwarm', fignum=1)
        plt.colorbar(cax)
        plt.xticks(range(len(self.numeric_cols)), self.numeric_cols, rotation=45, ha='left')
        plt.yticks(range(len(self.numeric_cols)), self.numeric_cols)
        plt.title('Correlation Heatmap (Numeric Features)', pad=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/02_correlation_heatmap.png')
        plt.close()

    def _plot_demographics(self):
        demo_cols = [c for c in self.numeric_cols if 'age' in c.lower() or 'bmi' in c.lower()]
        if not demo_cols: return
        
        fig, axes = plt.subplots(1, len(demo_cols), figsize=(15, 6))
        if len(demo_cols) == 1: axes = [axes]
        
        for ax, col in zip(axes, demo_cols):
            ax.hist(self.df[col], bins=20, color='#2ecc71', edgecolor='black')
            ax.set_title(f'{col} Distribution', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.grid(True, linestyle='--', alpha=0.6)
            
        plt.tight_layout()
        plt.savefig('outputs/03_demographics.png')
        plt.close()

    def _plot_lifestyle_factors(self):
        life_cols = [c for c in self.categorical_cols if any(k in c.lower() for k in ['smok', 'alcohol', 'activ', 'sleep'])]
        if not life_cols: return
        
        n_cols = len(life_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 6))
        if n_cols == 1: axes = [axes]
        
        for ax, col in zip(axes, life_cols):
            counts = self.df[col].value_counts()
            ax.bar(counts.index.astype(str), counts.values, color='#9b59b6')
            ax.set_title(f'{col}', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.savefig('outputs/04_lifestyle_factors.png')
        plt.close()

    def _plot_dietary_factors(self):
        diet_cols = [c for c in self.numeric_cols if any(k in c.lower() for k in ['meat', 'food', 'fiber', 'fruit', 'veg', 'water'])]
        if not diet_cols: return
        
        n_cols = len(diet_cols)
        fig, axes = plt.subplots(2, (n_cols + 1) // 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(diet_cols):
            axes[i].hist(self.df[col], bins=15, color='#f1c40f', edgecolor='black')
            axes[i].set_title(f'{col} Intake', fontweight='bold')
            
        for j in range(len(diet_cols), len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.savefig('outputs/05_dietary_factors.png')
        plt.close()

    def _plot_comparative_analysis(self):
        # Compare a key numeric feature against target
        key_num_cols = self.numeric_cols[:4] # Take first few for comparison
        if not key_num_cols: return
        
        fig, axes = plt.subplots(1, len(key_num_cols), figsize=(15, 6))
        if len(key_num_cols) == 1: axes = [axes]
        
        classes = self.df[self.target_column].unique()
        
        for ax, col in zip(axes, key_num_cols):
            data = [self.df[self.df[self.target_column] == cls][col].dropna() for cls in classes]
            ax.boxplot(data, labels=classes)
            ax.set_title(f'{col} by Target', fontweight='bold')
            ax.set_ylabel(col)
            ax.set_xlabel('Class')
            
        plt.tight_layout()
        plt.savefig('outputs/06_comparative_analysis.png')
        plt.close()

    # ==========================================
    # Phase 4: Feature Engineering
    # ==========================================
    def engineer_features(self):
        logging.info("--- Phase 4: Feature Engineering ---")
        
        # Example dynamic feature creation if age and bmi are present
        cols_lower = {c.lower(): c for c in self.df.columns}
        
        if 'age' in cols_lower and 'bmi' in cols_lower:
            logging.info("Engineering new feature: Age_BMI_Interaction")
            self.df['Age_BMI_Interaction'] = self.df[cols_lower['age']] * self.df[cols_lower['bmi']]
            self.numeric_cols.append('Age_BMI_Interaction')
            
        # Feature transformation - applying log transform to highly skewed numeric features
        for col in self.numeric_cols:
            if self.df[col].skew() > 1.5:
                logging.info(f"Applying log transformation to skewed feature: {col}")
                self.df[f'{col}_log'] = np.log1p(self.df[col])
                self.numeric_cols.append(f'{col}_log')
                self.numeric_cols.remove(col)
                self.df.drop(columns=[col], inplace=True)

        # Label encoding target if it's categorical
        if self.df[self.target_column].dtype == 'object':
            le = LabelEncoder()
            self.df[self.target_column] = le.fit_transform(self.df[self.target_column])
            logging.info("Label encoded the target variable.")

    # ==========================================
    # Phase 5: Machine Learning Pipeline Setup
    # ==========================================
    def setup_pipeline(self):
        logging.info("--- Phase 5: Machine Learning Pipeline Setup ---")
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ])
            
        logging.info("Pipeline preprocessor created.")

    # ==========================================
    # Phase 6 & 7: Train Multiple Models & Cross Validation
    # ==========================================
    def train_and_evaluate_models(self):
        logging.info("--- Phase 6 & 7: Model Training and Cross-Validation ---")
        
        classifiers = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'SVM': SVC(random_state=42, probability=True),
            'KNN': KNeighborsClassifier(),
            'Extra Trees': ExtraTreesClassifier(random_state=42)
        }

        results = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, clf in classifiers.items():
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                       ('classifier', clf)])
            
            # Cross validation
            cv_scores = cross_val_score(pipeline, self.X_train, self.y_train, cv=cv, scoring='accuracy')
            
            # Fit and test
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)
            
            acc = accuracy_score(self.y_test, y_pred)
            results.append({
                'Model': name,
                'CV Mean Accuracy': cv_scores.mean(),
                'CV Std': cv_scores.std(),
                'Test Accuracy': acc
            })
            self.models[name] = pipeline

        results_df = pd.DataFrame(results).sort_values(by='CV Mean Accuracy', ascending=False)
        print("\n--- Model Comparison ---")
        print(results_df.to_string(index=False))
        
        self.best_model_name = results_df.iloc[0]['Model']
        self.best_pipeline = self.models[self.best_model_name]
        logging.info(f"Best model selected: {self.best_model_name}")

    # ==========================================
    # Phase 8: Hyperparameter Tuning
    # ==========================================
    def hyperparameter_tuning(self):
        logging.info("--- Phase 8: Hyperparameter Tuning ---")
        
        if self.best_model_name == 'Random Forest':
            param_grid = {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5]
            }
        elif self.best_model_name == 'XGBoost':
            param_grid = {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__max_depth': [3, 5]
            }
        elif self.best_model_name == 'Logistic Regression':
            param_grid = {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__penalty': ['l2']
            }
        else:
            logging.info("Tuning skipped or default params used for this model to save time.")
            return

        grid_search = GridSearchCV(self.best_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        self.best_pipeline = grid_search.best_estimator_
        logging.info(f"Best parameters found: {grid_search.best_params_}")

    # ==========================================
    # Phase 9: Model Evaluation
    # ==========================================
    def evaluate_best_model(self):
        logging.info("--- Phase 9: Model Evaluation ---")
        
        y_pred = self.best_pipeline.predict(self.X_test)
        is_multiclass = len(np.unique(self.y_train)) > 2
        
        if hasattr(self.best_pipeline.named_steps['classifier'], "predict_proba"):
            y_prob = self.best_pipeline.predict_proba(self.X_test)
            if not is_multiclass:
                y_prob_plot = y_prob[:, 1]
            else:
                y_prob_plot = y_prob
        else:
            y_prob = y_pred
            y_prob_plot = y_pred

        avg_method = 'weighted' if is_multiclass else 'binary'
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, average=avg_method, zero_division=0),
            'Recall': recall_score(self.y_test, y_pred, average=avg_method, zero_division=0),
            'F1 Score': f1_score(self.y_test, y_pred, average=avg_method, zero_division=0),
        }
        
        try:
            if is_multiclass and hasattr(self.best_pipeline.named_steps['classifier'], "predict_proba"):
                metrics['ROC-AUC'] = roc_auc_score(self.y_test, y_prob, multi_class='ovr', average='weighted')
            else:
                metrics['ROC-AUC'] = roc_auc_score(self.y_test, y_prob_plot)
        except Exception:
            metrics['ROC-AUC'] = 0.0

        self.evaluation_results = metrics

        print("\n--- Best Model Metrics ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        # Visualizations
        self._plot_confusion_matrix(self.y_test, y_pred)
        if not is_multiclass:
            self._plot_roc_curve(self.y_test, y_prob_plot)
            self._plot_precision_recall_curve(self.y_test, y_prob_plot)
        else:
            logging.info("Skipping ROC and PR Curve plots for multiclass problem.")

    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 6))
        plt.matshow(cm, cmap='Blues', alpha=0.7, fignum=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix', pad=20, fontweight='bold')
        plt.savefig('outputs/07_confusion_matrix.png')
        plt.close()

    def _plot_roc_curve(self, y_true, y_prob):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic', fontweight='bold')
        plt.legend(loc="lower right")
        plt.savefig('outputs/08_roc_curve.png')
        plt.close()

    def _plot_precision_recall_curve(self, y_true, y_prob):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='purple', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve', fontweight='bold')
        plt.savefig('outputs/09_precision_recall_curve.png')
        plt.close()

    # ==========================================
    # Phase 10 & 11: Explainable AI & Risk Factor Analysis
    # ==========================================
    def explainable_ai(self):
        logging.info("--- Phase 10 & 11: Explainable AI & Risk Factor Analysis ---")
        
        # Transform X_train to get feature names
        X_train_transformed = self.preprocessor.fit_transform(self.X_train)
        
        # Get feature names
        num_features = self.numeric_cols
        
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        try:
            cat_features = list(cat_encoder.get_feature_names_out(self.categorical_cols))
        except:
            cat_features = [f"Cat_{i}" for i in range(X_train_transformed.shape[1] - len(num_features))]
            
        feature_names = num_features + cat_features
        
        classifier = self.best_pipeline.named_steps['classifier']
        
        # SHAP
        if isinstance(classifier, (RandomForestClassifier, XGBClassifier, DecisionTreeClassifier, GradientBoostingClassifier, ExtraTreesClassifier)):
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_train_transformed)
            
            # For tree explainers, shap_values might be a list for multi-class
            if hasattr(shap_values, "values"):
                sv = shap_values.values
            else:
                sv = shap_values
                
            if isinstance(sv, list):
                sv_plot = sv[1] # Use positive class
            else:
                sv_plot = sv
                
            if sv_plot.ndim == 3:
                sv_plot = sv_plot[:, :, 1]

            plt.figure()
            try:
                shap.summary_plot(sv_plot, X_train_transformed, feature_names=feature_names, show=False)
            except Exception as e:
                logging.warning(f"SHAP summary plot failed: {e}")
            plt.tight_layout()
            plt.savefig('outputs/10_shap_summary.png')
            plt.close()

            # Risk factors
            mean_abs_shap = np.abs(sv_plot).mean(axis=0)
            if mean_abs_shap.ndim > 1:
                mean_abs_shap = mean_abs_shap.mean(axis=1)
                
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': mean_abs_shap
            }).sort_values(by='Importance', ascending=False)
            
            print("\n--- Top Risk Factors Identified by SHAP ---")
            print(feature_importance.head(10).to_string(index=False))

            self.top_features = feature_importance.head(10)
        else:
            logging.info("SHAP is primarily configured here for tree-based models. Skipping SHAP plots for linear/kernel models to avoid heavy computation.")

    # ==========================================
    # Phase 12: Professional Dashboard-Style Outputs
    # ==========================================
    def create_dashboard(self):
        logging.info("--- Phase 12: Professional Dashboard Generation ---")
        # Compile multiple plots into a single dashboard image
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(2, 2, figure=fig)

        # ROC Curve
        ax1 = fig.add_subplot(gs[0, 0])
        is_multiclass = len(np.unique(self.y_train)) > 2
        if not is_multiclass:
            if hasattr(self.best_pipeline.named_steps['classifier'], "predict_proba"):
                y_prob = self.best_pipeline.predict_proba(self.X_test)[:, 1]
            else:
                y_prob = self.best_pipeline.predict(self.X_test)
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            ax1.plot(fpr, tpr, color='darkorange', lw=2)
            ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax1.set_title('ROC Curve')
        else:
            ax1.text(0.5, 0.5, 'ROC Curve\n(Not generated for multiclass)', 
                     horizontalalignment='center', verticalalignment='center')
            ax1.set_axis_off()
        
        # Confusion Matrix
        ax2 = fig.add_subplot(gs[0, 1])
        y_pred = self.best_pipeline.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        cax = ax2.matshow(cm, cmap='Blues')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')
        ax2.set_title('Confusion Matrix', pad=20)

        # Feature Importance (if applicable)
        if hasattr(self, 'top_features'):
            ax3 = fig.add_subplot(gs[1, :])
            ax3.barh(self.top_features['Feature'][::-1], self.top_features['Importance'][::-1], color='teal')
            ax3.set_title('Top 10 Influential Features (SHAP)')
            ax3.set_xlabel('Mean |SHAP Value|')
        
        plt.suptitle('Colorectal Cancer Prediction Dashboard', fontsize=24, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('outputs/11_final_dashboard.png')
        plt.close()

    # ==========================================
    # Phase 13: Final Report
    # ==========================================
    def generate_report(self):
        logging.info("--- Phase 13: Final Report Generation ---")
        
        report = f"""
=========================================================
COLORECTAL CANCER PREDICTION - FINAL PROJECT REPORT
=========================================================

1. EXECUTIVE SUMMARY
This project aims to predict the likelihood of colorectal cancer based on dietary, 
lifestyle, demographic, and health indicators. The entire data lifecycle from 
cleaning to advanced Explainable AI was implemented.

2. DATASET OVERVIEW
- Total Records Processed: {self.df.shape[0]}
- Total Features: {self.df.shape[1]}
- Target Variable: '{self.target_column}'

3. BEST MODEL PERFORMANCE
- Selected Model: {self.best_model_name}
- Accuracy:  {self.evaluation_results.get('Accuracy', 0):.4f}
- Precision: {self.evaluation_results.get('Precision', 0):.4f}
- Recall:    {self.evaluation_results.get('Recall', 0):.4f}
- F1 Score:  {self.evaluation_results.get('F1 Score', 0):.4f}
- ROC-AUC:   {self.evaluation_results.get('ROC-AUC', 0):.4f}

4. KEY RISK FACTORS IDENTIFIED
Based on SHAP Analysis, the top features driving predictions were:
"""
        if hasattr(self, 'top_features'):
            for idx, row in self.top_features.iterrows():
                report += f" - {row['Feature']} (Importance: {row['Importance']:.4f})\n"
        else:
            report += " (Feature importance analysis not applicable for chosen model class)\n"

        report += """
5. LIMITATIONS & FUTURE IMPROVEMENTS
- Limitations: Reliance on potentially biased or synthetic sampling, feature assumptions.
- Future Work: Deep learning implementation, longitudinal patient data integration, 
  and external clinical validation.

=========================================================
"""
        with open('outputs/12_final_report.txt', 'w') as f:
            f.write(report)
        print(report)

def main():
    # Example usage:
    # Set path to the dataset. If it doesn't exist, the script generates a robust synthetic one.
    DATASET_PATH = 'crc_dataset.csv'
    
    project = ColorectalCancerProject(data_path=DATASET_PATH)
    
    project.load_and_inspect_data()
    project.clean_data()
    project.perform_eda()
    project.engineer_features()
    project.setup_pipeline()
    project.train_and_evaluate_models()
    project.hyperparameter_tuning()
    project.evaluate_best_model()
    project.explainable_ai()
    project.create_dashboard()
    project.generate_report()
    
    logging.info("Project executed successfully! Check the 'outputs' directory for charts and the final report.")

if __name__ == "__main__":
    main()
