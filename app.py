import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

def load_data():
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    data = pd.read_csv(url)
    kenya = data[data['country'] == 'Kenya']
    kenya = kenya[['year', 'gdp', 'population', 'energy_per_capita', 'co2']].dropna()
    kenya['energy_use'] = kenya['energy_per_capita'] * kenya['population']
    kenya['gdp_pc'] = kenya['gdp'] / kenya['population']
    print(f"Loaded {len(kenya)} rows ({kenya['year'].min()}–{kenya['year'].max()})")
    return kenya

def get_features_targets(df):
    X = df[['gdp', 'population', 'energy_use', 'gdp_pc']]
    y = df['co2']
    return X, y

def run_models(X_train, X_test, y_train, y_test):
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(alpha=0.1),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    scores = {}
    preds = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ('scale', StandardScaler()),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        cv = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')

        scores[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'CV_Mean': cv.mean(),
            'CV_Std': cv.std()
        }
        preds[name] = y_pred

    return scores, preds

def make_plots(df, X, y, scores, preds, y_test):
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))

    ax[0, 0].plot(df['year'], df['co2'], marker='o', linestyle='-', color='blue')
    ax[0, 0].set_title('Kenya CO₂ Emissions Over Time')

    corr = X.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax[0, 1])
    ax[0, 1].set_title('Feature Correlation')

    r2_vals = [scores[m]['R2'] for m in scores]
    ax[0, 2].bar(scores.keys(), r2_vals, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax[0, 2].set_title('R² by Model')
    ax[0, 2].set_ylim(0, 1)

    best = max(scores, key=lambda x: scores[x]['R2'])
    best_pred = preds[best]

    ax[1, 0].scatter(y_test, best_pred, alpha=0.7)
    ax[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax[1, 0].set_title(f'Actual vs Predicted ({best})')

    res = y_test - best_pred
    ax[1, 1].scatter(best_pred, res, alpha=0.7)
    ax[1, 1].axhline(0, color='red', linestyle='--')
    ax[1, 1].set_title('Residual Plot')

    if 'RandomForest' in scores:
        rf = Pipeline([
            ('scale', StandardScaler()),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        rf.fit(X, y)
        imp = rf.named_steps['rf'].feature_importances_
        idx = np.argsort(imp)
        ax[1, 2].barh(X.columns[idx], imp[idx], color='grey')
        ax[1, 2].set_title('Random Forest Feature Importance')

    plt.tight_layout()
    plt.savefig('comprehensive_co2_analysis.png', dpi=300)
    plt.show()

def print_results(scores):
    print("\nModel Results")
    for model, metrics in scores.items():
        print(f"{model}: R²={metrics['R2']:.3f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, CV={metrics['CV_Mean']:.3f} ± {metrics['CV_Std']:.3f}")
    best = max(scores, key=lambda x: scores[x]['R2'])
    print(f"\nBest model: {best}")

def main():
    df = load_data()
    X, y = get_features_targets(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    scores, preds = run_models(X_train, X_test, y_train, y_test)
    make_plots(df, X, y, scores, preds, y_test)
    print_results(scores)

    print("\nNote: These predictions depend on historic data patterns only.")

if __name__ == "__main__":
    main()
