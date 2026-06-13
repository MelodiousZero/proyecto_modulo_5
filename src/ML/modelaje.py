from EDA.clean_dataset import cleanDataFrame
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


class compareModels:
    def __init__(self):

        self.dataframe = cleanDataFrame().run_clean()
        self.dependent_variable = "Rented Bike Count"
        self.indepent_variables = list(set(self.dataframe.columns.tolist())-{self.dependent_variable})
        self.y = np.sqrt(self.dataframe[self.dependent_variable])
        self.X = self.dataframe.drop("Rented Bike Count",axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 0) #antes de usar self.scaler para evitar data leakage
        self.columns_table = ['Model', 'MAE', 'MSE', 'RMSE', 'R2_score', 'Adjusted_R2']
        self.comparison_table = pd.DataFrame(columns=self.columns_table)
        self.alphas = np.logspace(-4,2,50)
        self.l1_ratios = np.linspace(0.1, 1.0, 6)
        self.grid_search_comparison_table = pd.DataFrame(columns=self.columns_table)
        self.seed = 97
        self.scaler = StandardScaler()

    
    def linear_regression(self):
        regressor = LinearRegression()
        regressor.fit(self.X_train, self.y_train)
        y_pred_train = regressor.predict(self.X_train)

        y_pred = regressor.predict(self.X_test)
        return y_pred,regressor
         

    def lasso(self,alpha=0.01):
        lasso_regressor = Lasso(alpha=alpha)
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        lasso_regressor.fit(X_train_scaled,self.y_train)

        y_pred_train = lasso_regressor.predict(X_train_scaled)
        y_pred = lasso_regressor.predict(X_test_scaled)
        return y_pred,lasso_regressor

    def ridge(self,alpha=0.01):
        ridge_regressor = Ridge(alpha=alpha)
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        ridge_regressor.fit(X_train_scaled,self.y_train)

        y_pred_train = ridge_regressor.predict(X_train_scaled)
        y_pred = ridge_regressor.predict(X_test_scaled)
        return y_pred,ridge_regressor

    
    def elastic_net(self,alpha=0.001,l1_ratio=0.3):
        elastic_net_regressor = ElasticNet(alpha = alpha,l1_ratio = l1_ratio)
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        elastic_net_regressor.fit(X_train_scaled,self.y_train)
        
        y_pred_train = elastic_net_regressor.predict(X_train_scaled)
        y_pred = elastic_net_regressor.predict(X_test_scaled)
        return y_pred,elastic_net_regressor
    
    def poisson_GLM(self):
        poisson_sklearn = make_pipeline(
            self.scaler, 
            PoissonRegressor(alpha=0.0, max_iter=1000) 
        )

        poisson_sklearn.fit(self.X_train, self.y_train)
        y_pred = poisson_sklearn.predict(self.X_test)
        return y_pred,poisson_sklearn
    
    def decision_tree(self,max_depth=9,min_samples_leaf=3,min_samples_split=15):
        decision_tree_regressor = DecisionTreeRegressor(max_depth=max_depth,
                                                        min_samples_leaf=min_samples_leaf,
                                                        min_samples_split=min_samples_split)
        decision_tree_regressor.fit(self.X_train,self.y_train)
        y_pred_train = decision_tree_regressor.predict(self.X_train)
        y_pred = decision_tree_regressor.predict(self.X_test)
        return y_pred,decision_tree_regressor
    
    def random_forest(self,max_depth=None,min_samples_split=3,n_estimators=150):
        random_forest_regressor = RandomForestRegressor(max_depth=max_depth,
                                                        min_samples_split=min_samples_split,
                                                        n_estimators=n_estimators)
        random_forest_regressor.fit(self.X_train,self.y_train)
        y_pred_train = random_forest_regressor.predict(self.X_train)
        y_pred = random_forest_regressor.predict(self.X_test)
        return y_pred,random_forest_regressor

    
  
    def gradient_boosting(self,learning_rate=0.1,max_depth=4,n_estimators=200,loss="squared_error"):
        gradient_boosting_regressor = GradientBoostingRegressor(learning_rate=learning_rate,
                                                                max_depth=max_depth,
                                                                n_estimators=n_estimators,
                                                                loss=loss)
        gradient_boosting_regressor.fit(self.X_train,self.y_train)
        y_pred_train = gradient_boosting_regressor.predict(self.X_train)
        y_pred = gradient_boosting_regressor.predict(self.X_test)
        return y_pred,gradient_boosting_regressor
    
    

    def ada_boost(self, learning_rate=1.0, n_estimators=50, loss='linear'):
        ada_boost_regressor = AdaBoostRegressor(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            loss=loss,
            random_state=self.seed
        )
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        ada_boost_regressor.fit(X_train_scaled, self.y_train)
        y_pred = ada_boost_regressor.predict(X_test_scaled)
        return y_pred,ada_boost_regressor

    def xg_boost(self, learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.8,objective="reg:squarederror"):
        xgboost_regressor = XGBRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            subsample=subsample,
            random_state=self.seed,
            verbosity=0,
            objective = objective 
        )
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        xgboost_regressor.fit(X_train_scaled, self.y_train)
        y_pred = xgboost_regressor.predict(X_test_scaled)
        return y_pred,xgboost_regressor

    def mlp_regressor(self, hidden_layer_sizes=(100,), activation='relu', alpha=0.0001, learning_rate='constant'):
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate=learning_rate,
            max_iter=1000,
            random_state=self.seed,
            early_stopping=True,
            validation_fraction=0.1
        )
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        mlp.fit(X_train_scaled, self.y_train)
        y_pred = mlp.predict(X_test_scaled)
        return y_pred,mlp
        

    

    
    def grid_searching(self,scoring="r2"):
        #Escalamiento variables.
        X_train_scaled = self.scaler.fit_transform(self.X_train)

        #lasso
        lasso_param_grid = {"alpha":self.alphas}
        lasso = Lasso(max_iter=10000)
        lasso_grid = GridSearchCV(lasso,lasso_param_grid,cv=5,scoring="r2")
        lasso_grid.fit(X_train_scaled, self.y_train)
        #ridge
        ridge_param_grid = {"alpha":self.alphas}
        ridge = Ridge(max_iter=10000)
        ridge_grid = GridSearchCV(ridge,ridge_param_grid,cv=5,scoring="r2")
        ridge_grid.fit(X_train_scaled, self.y_train)
        #elastic net
        elastic_net_paragram_grid = {
            "alpha": self.alphas,
            "l1_ratio": self.l1_ratios
        }
        elastic_net = ElasticNet(max_iter=10000)
        elastic_net_grid = GridSearchCV(elastic_net,elastic_net_paragram_grid,cv=5,scoring="r2")
        elastic_net_grid.fit(X_train_scaled, self.y_train)
        #decision tree
        decision_tree_param_grid = {
            'max_depth': [3, 5, 7, 9, 11, 15, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10]
        }
        decision_tree = DecisionTreeRegressor(random_state=self.seed)
        decision_tree_grid = GridSearchCV(decision_tree,decision_tree_param_grid,cv=5,scoring="r2")
        decision_tree_grid.fit(self.X_train,self.y_train)
        #random forest
        random_forest_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
        random_forest = RandomForestRegressor(random_state=self.seed)
        random_forest_grid = GridSearchCV(random_forest,random_forest_param_grid,cv=5,scoring="r2")
        random_forest_grid.fit(self.X_train,self.y_train)
        #gradient_boosting
        gradient_boosting_param_grid = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7]
        }
        gradient_boosting = GradientBoostingRegressor(random_state=self.seed)
        gradient_boosting_grid = GridSearchCV(gradient_boosting,gradient_boosting_param_grid,cv=5,scoring="r2")
        gradient_boosting_grid.fit(self.X_train,self.y_train)

        #AdaBoost
        ada_param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.5, 1.0, 2.0],
            'loss': ['linear', 'square', 'exponential']
        }
        ada = AdaBoostRegressor(random_state=self.seed)
        ada_grid = GridSearchCV(ada, ada_param_grid, cv=5, scoring='r2')
        ada_grid.fit(X_train_scaled, self.y_train)

        #XGBoost
        xgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0]
        }
        xgb = XGBRegressor(random_state=self.seed, verbosity=0)
        xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=5, scoring='r2')
        xgb_grid.fit(X_train_scaled, self.y_train)

        #MLP
        mlp_param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 25)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        mlp = MLPRegressor(max_iter=1000, random_state=self.seed, early_stopping=True)
        mlp_grid = GridSearchCV(mlp, mlp_param_grid, cv=5, scoring='r2')
        mlp_grid.fit(X_train_scaled, self.y_train)

        best_params = {
            "lasso":lasso_grid,
            "ridge": ridge_grid,
            "elastic_net": elastic_net_grid,
            "decision_tree":decision_tree_grid,
            "random_forest": random_forest_grid,
            "gradient_boost": gradient_boosting_grid,
            "ada_boost": ada_grid,           
            "xg_boost": xgb_grid,            
            "mlp": mlp_grid   
        }

        print(best_params)

        return best_params
    
    def re_run_with_better_params(self,best_params=None):
        if not best_params:
            og_params = self.grid_searching()
            best_params = {
            name: gs.best_params_
            for name, gs in og_params.items()
        }
        else:
            og_params = best_params    

            
        #Lasso
        lasso_y_pred,lasso = self.lasso(best_params["lasso"]["alpha"])
        self.create_entry(lasso_y_pred,"Lasso (Grid Search)",self.grid_search_comparison_table)
        
        #Ridge
        ridge_y_pred,ridge = self.ridge(best_params["ridge"]["alpha"])
        self.create_entry(ridge_y_pred,"Ridge (Grid Search)",self.grid_search_comparison_table)

        #Elastic Net
        elastic_net_y_pred,elastic_net = self.elastic_net(best_params["elastic_net"]["alpha"],best_params["elastic_net"]["l1_ratio"])
        self.create_entry(elastic_net_y_pred,"Elastic Net (Grid Search)",self.grid_search_comparison_table)

        #Decision Tree
        decision_tree_y_pred,decision_tree = self.decision_tree(best_params["decision_tree"]["max_depth"],
                                                  best_params["decision_tree"]["min_samples_leaf"],
                                                  best_params["decision_tree"]["min_samples_split"])
        
        self.create_entry(decision_tree_y_pred,"Decision Tree (Grid Search)",self.grid_search_comparison_table)
        
        #Random Forest
        random_forest_y_pred ,random_forest= self.random_forest(best_params["random_forest"]["max_depth"],
                                                  best_params["random_forest"]["min_samples_split"],
                                                  best_params["random_forest"]["n_estimators"])
        
        self.create_entry(random_forest_y_pred,"Random Forest (Grid Search)",self.grid_search_comparison_table)
        
        #Gradient Boost
        gradient_boost_y_pred,gradient_boost = self.gradient_boosting(best_params["gradient_boost"]["learning_rate"],
                                                       best_params["gradient_boost"]["max_depth"],
                                                       best_params["gradient_boost"]["n_estimators"])
        self.create_entry(gradient_boost_y_pred,"Gradient Boost (Grid Search)",self.grid_search_comparison_table)


        #AdaBoost
        ada_y_pred,ada_boost = self.ada_boost(
            learning_rate=best_params["ada_boost"]["learning_rate"],
            n_estimators=best_params["ada_boost"]["n_estimators"],
            loss=best_params["ada_boost"]["loss"]
        )
        self.create_entry(ada_y_pred, "AdaBoost (Grid Search)", self.grid_search_comparison_table)

        #XGBoost
        xgb_y_pred,xg_boost = self.xg_boost(
            learning_rate=best_params["xg_boost"]["learning_rate"],
            max_depth=best_params["xg_boost"]["max_depth"],
            n_estimators=best_params["xg_boost"]["n_estimators"],
            subsample=best_params["xg_boost"]["subsample"]
        )
        self.create_entry(xgb_y_pred, "XGBoost (Grid Search)", self.grid_search_comparison_table)

        #MLP
        mlp_y_pred,mlp = self.mlp_regressor(
            hidden_layer_sizes=best_params["mlp"]["hidden_layer_sizes"],
            activation=best_params["mlp"]["activation"],
            alpha=best_params["mlp"]["alpha"],
            learning_rate=best_params["mlp"]["learning_rate"]
        )
        self.create_entry(mlp_y_pred, "MLP (Grid Search)", self.grid_search_comparison_table)

        return {"lasso": lasso_y_pred,
                "ridge":ridge_y_pred,
                "elastic_net":elastic_net_y_pred,
                "decision_tree":decision_tree_y_pred,
                "random_forest":random_forest_y_pred,
                "gradient_boost":gradient_boost_y_pred,
                "ada_boost":ada_y_pred,
                "xgboost":xgb_y_pred,
                "mlp":mlp_y_pred
            }, best_params
        

   

        
    def most_important_features(self):
        important_variables = self.X.columns
        gradient_boosting = GradientBoostingRegressor(learning_rate=0.1, max_depth=7, n_estimators=300)
        gradient_boosting.fit(self.X_train, self.y_train)
        importances = gradient_boosting.feature_importances_
        indices = np.argsort(importances)

        plt.figure(figsize=(12,6))
        plt.title('Feature Importances (Gradient Boosting - GridSearchCV)', fontsize=14)

        bar_colors = ['#CD2E3A' if i % 2 == 0 else '#0047A0' for i in range(len(indices))]

        plt.barh(range(len(indices)), importances[indices], color=bar_colors, align='center')
        plt.yticks(range(len(indices)), important_variables[indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.show()


    
    def create_entry(self,y_pred,model_name,datafame):
        MAE = mean_absolute_error(self.y_test,y_pred)

        MSE = mean_squared_error(self.y_test,y_pred)

        RMSE = np.sqrt(MSE)

        R2 = r2_score(self.y_test,y_pred)

        adj_r2 = 1-(1-r2_score(self.y_test,y_pred))*((self.X_test.shape[0]-1)/(self.X_test.shape[0]-self.X_test.shape[1]-1))

        test_dict = {'Model':model_name,
              'MAE':round(MAE,4),
              'MSE':round(MSE,4),
              'RMSE':round(RMSE,4),
              'R2_score':round(R2,4),
              'Adjusted_R2':round(adj_r2,4)}
        datafame.loc[len(datafame)] = test_dict
        return datafame
    


    def plot_results(self,y_pred, model_name, figsize=(8, 6), fontsize=12,save_path=None):
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        residuals = self.y_test - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        ax1 = axes[0, 0]
        ax1.scatter(self.y_test, y_pred, alpha=0.6, edgecolors='w', s=70, color='#2E86AB')
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Values', fontsize=fontsize)
        ax1.set_ylabel('Predicted Values', fontsize=fontsize)
        ax1.set_title('Predicted vs Actual', fontsize=fontsize, fontweight='bold')
        textstr = f'$R^2 = {r2:.3f}$\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=fontsize,
                verticalalignment='top', bbox=props)
        ax1.legend(loc='lower right')
        
        ax2 = axes[0, 1]
        sns.regplot(x=y_pred, y=residuals, ax=ax2, line_kws={'color': 'red'}, scatter_kws={'alpha':0.5})
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.set_xlabel('Predicted Values', fontsize=fontsize)
        ax2.set_ylabel('Residuals', fontsize=fontsize)
        ax2.set_title('Residuals vs Predicted', fontsize=fontsize, fontweight='bold')
        
        ax3 = axes[1, 0]
        sns.histplot(residuals, kde=True, ax=ax3, color='#2E86AB', bins=30)
        ax3.set_xlabel('Residuals', fontsize=fontsize)
        ax3.set_ylabel('Frequency', fontsize=fontsize)
        ax3.set_title('Residual Distribution', fontsize=fontsize, fontweight='bold')
        
        ax4 = axes[1, 1]
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot', fontsize=fontsize, fontweight='bold')
        ax4.get_lines()[0].set_marker('o')
        ax4.get_lines()[0].set_markersize(4)
        ax4.get_lines()[0].set_alpha(0.6)
        ax4.get_lines()[1].set_color('red')
        ax4.get_lines()[1].set_linewidth(2)
        
        fig.suptitle(f"{model_name}",fontsize=fontsize,fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_coefficients(self, model, model_name):
        if hasattr(model, 'coef_'):
            coefs = model.coef_
        elif hasattr(model, 'named_steps'):  # pipeline
            coefs = model.named_steps['poissonregressor'].coef_
        else:
            print("No coefficients found")
            return
        features = self.X.columns
        plt.figure(figsize=(10,5))
        plt.bar(features, coefs)
        plt.xticks(rotation=90)
        plt.title(f'Coeficientes – {model_name}')
        plt.tight_layout()
        plt.show()

