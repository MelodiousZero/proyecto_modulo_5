from EDA.clean_dataset import cleanDataFrame
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
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


class compareModels:
    def __init__(self):

        self.dataframe = cleanDataFrame().run_clean()
        self.dependent_variable = cleanDataFrame().target
        self.indepent_variables = list(set(self.dataframe.columns.tolist())-{"Rented Bike Count"})
        self.y = np.sqrt(self.dataframe["Rented Bike Count"])
        self.X = self.dataframe.drop("Rented Bike Count",axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 0)
        self.columns_table = ['Model', 'MAE', 'MSE', 'RMSE', 'R2_score', 'Adjusted_R2']
        self.comparison_table = pd.DataFrame(columns=self.columns_table)
        self.alphas = np.logspace(-4,2,50)
        self.l1_ratios = np.linspace(0.1, 1.0, 6)
        self.grid_search_comparison_table = pd.DataFrame(columns=self.columns_table)
        self.seed = 97

    
    def linear_regression(self):
        regressor = LinearRegression()
        regressor.fit(self.X_train, self.y_train)
        y_pred_train = regressor.predict(self.X_train)

        y_pred = regressor.predict(self.X_test)
        return y_pred
         

    def lasso(self,alpha=0.01):
        lasso_regressor = Lasso(alpha=alpha)
        lasso_regressor.fit(self.X_train,self.y_train)
        y_pred_train = lasso_regressor.predict(self.X_train)
        y_pred = lasso_regressor.predict(self.X_test)
        return y_pred

    def ridge(self,alpha=0.01):
        ridge_regressor = Ridge(alpha=alpha)
        ridge_regressor.fit(self.X_train,self.y_train)
        y_pred_train = ridge_regressor.predict(self.X_train)
        y_pred = ridge_regressor.predict(self.X_test)
        return y_pred

    
    def elastic_net(self,alpha=0.001,l1_ratio=0.3):
        elastic_net_regressor = ElasticNet(alpha = alpha,l1_ratio = l1_ratio)
        elastic_net_regressor.fit(self.X_train,self.y_train)
        y_pred_train = elastic_net_regressor.predict(self.X_train)
        y_pred = elastic_net_regressor.predict(self.X_test)
        return y_pred
    
    def decision_tree(self,max_depth=9,min_samples_leaf=3,min_samples_split=15):
        decision_tree_regressor = DecisionTreeRegressor(max_depth=max_depth,
                                                        min_samples_leaf=min_samples_leaf,
                                                        min_samples_split=min_samples_split)
        decision_tree_regressor.fit(self.X_train,self.y_train)
        y_pred_train = decision_tree_regressor.predict(self.X_train)
        y_pred = decision_tree_regressor.predict(self.X_test)
        return y_pred
    
    def random_forest(self,max_depth=None,min_samples_split=3,n_estimators=150):
        random_forest_regressor = RandomForestRegressor(max_depth=max_depth,
                                                        min_samples_split=min_samples_split,
                                                        n_estimators=n_estimators)
        random_forest_regressor.fit(self.X_train,self.y_train)
        y_pred_train = random_forest_regressor.predict(self.X_train)
        y_pred = random_forest_regressor.predict(self.X_test)
        return y_pred

    
  
    def gradient_boosting(self,learning_rate=0.1,max_depth=4,n_estimators=200):
        gradient_boosting_regressor = GradientBoostingRegressor(learning_rate=learning_rate,
                                                                max_depth=max_depth,
                                                                n_estimators=n_estimators)
        gradient_boosting_regressor.fit(self.X_train,self.y_train)
        y_pred_train = gradient_boosting_regressor.predict(self.X_train)
        y_pred = gradient_boosting_regressor.predict(self.X_test)
        return y_pred
    

    
    def grid_searching(self):
        #lasso
        lasso_param_grid = {"alpha":self.alphas}
        lasso = Lasso(max_iter=100000)
        lasso_grid = GridSearchCV(lasso,lasso_param_grid,cv=5,scoring="r2")
        lasso_grid.fit(self.X_train, self.y_train)
        best_lasso_alpha = lasso_grid.best_params_['alpha']
        #ridge
        ridge_param_grid = {"alpha":self.alphas}
        ridge = Ridge(max_iter=100000)
        ridge_grid = GridSearchCV(ridge,ridge_param_grid,cv=5,scoring="r2")
        ridge_grid.fit(self.X_train, self.y_train)
        best_ridge_alpha = ridge_grid.best_params_['alpha']
        #elastic net
        elastic_net_paragram_grid = {
            "alpha": self.alphas,
            "l1_ratio": self.l1_ratios
        }
        elastic_net = ElasticNet(max_iter=100000)
        elastic_net_grid = GridSearchCV(elastic_net,elastic_net_paragram_grid,cv=5,scoring="r2")
        elastic_net_grid.fit(self.X_train, self.y_train)
        best_elastic_net_params = elastic_net_grid.best_params_
        #decision tree
        decision_tree_param_grid = {
            'max_depth': [3, 5, 7, 9, 11, 15, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10]
        }
        decision_tree = DecisionTreeRegressor(random_state=self.seed)
        decision_tree_grid = GridSearchCV(decision_tree,decision_tree_param_grid,cv=5,scoring="r2")
        decision_tree_grid.fit(self.X_train,self.y_train)
        best_decision_tree_params = decision_tree_grid.best_params_
        #random forest
        random_forest_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
        random_forest = RandomForestRegressor(random_state=self.seed)
        random_forest_grid = GridSearchCV(random_forest,random_forest_param_grid,cv=5,scoring="r2")
        random_forest_grid.fit(self.X_train,self.y_train)
        best_random_forest_params = random_forest_grid.best_params_
        #gradient_boosting
        gradient_boosting_param_grid = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7]
        }
        gradient_boosting = GradientBoostingRegressor(random_state=self.seed)
        gradient_boosting_grid = GridSearchCV(gradient_boosting,gradient_boosting_param_grid,cv=5,scoring="r2")
        gradient_boosting_grid.fit(self.X_train,self.y_train)
        best_gradient_boosting_params=gradient_boosting_grid.best_params_

        best_params = {
            "lasso":best_lasso_alpha,
            "ridge": best_ridge_alpha,
            "elastic_net": best_elastic_net_params,
            "decision_tree":best_decision_tree_params,
            "random_forest": best_random_forest_params,
            "gradient_boost": best_gradient_boosting_params
        }

        print(best_params)

        return best_params
    
    def re_run_with_better_params(self):
        best_params = self.grid_searching()

        lasso_y_pred = self.lasso(best_params["lasso"])

        self.create_entry(lasso_y_pred,"Lasso (Grid Search)",self.grid_search_comparison_table)

        ridge_y_pred = self.ridge(best_params["ridge"])

        self.create_entry(ridge_y_pred,"Ridge (Grid Search)",self.grid_search_comparison_table)


        elastic_net_y_pred = self.elastic_net(best_params["elastic_net"]["alpha"],best_params["elastic_net"]["l1_ratio"])

        self.create_entry(elastic_net_y_pred,"Elastic Net (Grid Search)",self.grid_search_comparison_table)

        decision_tree_y_pred = self.decision_tree(best_params["decision_tree"]["max_depth"],
                                                  best_params["decision_tree"]["min_samples_leaf"],
                                                  best_params["decision_tree"]["min_samples_split"])
        
        self.create_entry(decision_tree_y_pred,"Decision Tree (Grid Search)",self.grid_search_comparison_table)
        
        random_forest_y_pred = self.random_forest(best_params["random_forest"]["max_depth"],
                                                  best_params["random_forest"]["min_samples_split"],
                                                  best_params["random_forest"]["n_estimators"])
        
        self.create_entry(random_forest_y_pred,"Random Forest (Grid Search)",self.grid_search_comparison_table)
        
        gradient_boost_y_pred = self.gradient_boosting(best_params["gradient_boost"]["learning_rate"],
                                                       best_params["gradient_boost"]["max_depth"],
                                                       best_params["gradient_boost"]["n_estimators"])
        self.create_entry(gradient_boost_y_pred,"Gradient Boost (Grid Search)",self.grid_search_comparison_table)

        return {"lasso": lasso_y_pred,"ridge":ridge_y_pred,"elastic_net":elastic_net_y_pred,
                "decision_tree":decision_tree_y_pred,"random_forest":random_forest_y_pred,"gradient_boost":gradient_boost_y_pred}
        


        
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
        print(f"The Mean Absolute Error (MAE) is {MAE}.")

        MSE = mean_squared_error(self.y_test,y_pred)
        print(f"The Mean Squred Error(MSE) is {MSE}.")

        RMSE = np.sqrt(MSE)
        print(f"The Root Mean Squared Error(RMSE) is {RMSE}.")

        R2 = r2_score(self.y_test,y_pred)
        print(f"The R2 Score is {R2}.")

        adj_r2 = 1-(1-r2_score(self.y_test,y_pred))*((self.X_test.shape[0]-1)/(self.X_test.shape[0]-self.X_test.shape[1]-1))
        print(f"Adjusted R2 is {adj_r2}.")

        test_dict = {'Model':model_name,
              'MAE':round(MAE,4),
              'MSE':round(MSE,4),
              'RMSE':round(RMSE,4),
              'R2_score':round(R2,4),
              'Adjusted_R2':round(adj_r2,4)}
        datafame.loc[len(datafame)] = test_dict
    


    def plot_results(self,y_pred, model_name, figsize=(8, 6), save_path=None):
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
        ax1.set_xlabel('Actual Values', fontsize=11)
        ax1.set_ylabel('Predicted Values', fontsize=11)
        ax1.set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
        # Add metrics text
        textstr = f'$R^2 = {r2:.3f}$\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        ax1.legend(loc='lower right')
        
        ax2 = axes[0, 1]
        sns.regplot(x=y_pred, y=residuals, ax=ax2, line_kws={'color': 'red'}, scatter_kws={'alpha':0.5})
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.set_xlabel('Predicted Values', fontsize=11)
        ax2.set_ylabel('Residuals', fontsize=11)
        ax2.set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
        
        ax3 = axes[1, 0]
        sns.histplot(residuals, kde=True, ax=ax3, color='#2E86AB', bins=30)
        ax3.set_xlabel('Residuals', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        
        ax4 = axes[1, 1]
        import scipy.stats as stats
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        ax4.get_lines()[0].set_marker('o')
        ax4.get_lines()[0].set_markersize(4)
        ax4.get_lines()[0].set_alpha(0.6)
        ax4.get_lines()[1].set_color('red')
        ax4.get_lines()[1].set_linewidth(2)
        
        fig.suptitle(f"{model_name}",fontsize=16,fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
            
