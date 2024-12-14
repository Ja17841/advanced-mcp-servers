from fastmcp import MCPServer, Resource
import pandas as pd
import numpy as np
from scipy import stats
import json

class DataAnalysisServer(MCPServer):
    """MCP server for data analysis using pandas and numpy"""
    
    @Resource
    async def basic_analysis(self, data: dict) -> dict:
        """Perform basic statistical analysis on the data"""
        df = pd.DataFrame(data)
        
        analysis = {
            "summary": df.describe().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "shape": df.shape
        }
        
        return analysis
    
    @Resource
    async def correlation_analysis(self, data: dict, method: str = "pearson") -> dict:
        """Calculate correlation matrix using specified method"""
        df = pd.DataFrame(data)
        numeric_df = df.select_dtypes(include=[np.number])
        
        return {
            "correlation_matrix": numeric_df.corr(method=method).to_dict(),
            "numeric_columns": numeric_df.columns.tolist()
        }
    
    @Resource
    async def statistical_tests(self, data: dict, test_type: str, **kwargs) -> dict:
        """Perform statistical tests on the data"""
        df = pd.DataFrame(data)
        results = {}
        
        if test_type == "normality":
            for column in df.select_dtypes(include=[np.number]).columns:
                stat, p_value = stats.normaltest(df[column].dropna())
                results[column] = {
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "is_normal": p_value > 0.05
                }
        
        elif test_type == "ttest":
            col1, col2 = kwargs.get("columns", [])
            stat, p_value = stats.ttest_ind(
                df[col1].dropna(),
                df[col2].dropna()
            )
            results = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05
            }
        
        return results
    
    @Resource
    async def feature_engineering(self, data: dict, operations: list) -> dict:
        """Perform feature engineering operations"""
        df = pd.DataFrame(data)
        results = {}
        
        for op in operations:
            if op["type"] == "log":
                col = op["column"]
                results[f"{col}_log"] = np.log1p(df[col]).tolist()
            
            elif op["type"] == "standardize":
                col = op["column"]
                results[f"{col}_standardized"] = stats.zscore(df[col]).tolist()
            
            elif op["type"] == "bin":
                col = op["column"]
                bins = op.get("bins", 10)
                results[f"{col}_binned"] = pd.qcut(
                    df[col],
                    q=bins,
                    labels=[f"bin_{i}" for i in range(bins)]
                ).astype(str).tolist()
        
        return results

if __name__ == "__main__":
    server = DataAnalysisServer()
    server.run(host="0.0.0.0", port=8001)