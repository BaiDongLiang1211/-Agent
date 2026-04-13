import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from config import DATA_FILENAME

# 载入通用数据源
df = pd.read_csv(DATA_FILENAME)

def get_data_summary():
    """获取数据集的通用统计摘要。"""
    numeric_df = df.select_dtypes(include=['number'])
    # 转换为整数或保留两位小数的字符串
    summary = numeric_df.describe().round(2).T
    summary['variance'] = numeric_df.var().round(2)
    return summary.to_string()

def create_distribution_plot(column_name: str, title: str = "Distribution Plot"):
    """
    通用绘图工具：绘制指定列的分布图。
    :param column_name: 要统计的列名
    :param title: 图表标题
    """
    if column_name not in df.columns:
        return f"Error: 列名 '{column_name}' 不存在。"
    
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column_name, data=df)
    plt.title(title)
    save_path = f"{column_name}_distribution.png"
    plt.savefig(save_path)
    plt.close()
    return f"Success: 图表已保存至 {save_path}"

def run_sklearn_classification(target_column: str, feature_columns: list):
    """
    通用机器学习工具：执行分类任务。
    :param target_column: 预测的目标列名
    :param feature_columns: 用作特征的列名列表
    """
    try:
        data = df.copy()
        # 自动处理缺失值
        for col in feature_columns:
            if data[col].dtype == 'object':
                data[col] = LabelEncoder().fit_transform(data[col].astype(str))
            data[col] = data[col].fillna(data[col].median() if data[col].dtype != 'object' else data[col].mode()[0])
        
        X = data[feature_columns]
        y = data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        return f"针对 {target_column} 的模型训练完成，测试集准确率：{acc:.2%}"
    except Exception as e:
        return f"Error: 建模失败 - {str(e)}"

# 工具映射
available_functions = {
    "get_data_summary": get_data_summary,
    "create_distribution_plot": create_distribution_plot,
    "run_sklearn_classification": run_sklearn_classification
}

# 通用工具 Schema 定义
tools_config = [
    {
        "type": "function",
        "function": {
            "name": "get_data_summary",
            "description": "对数据集进行基础统计摘要分析"
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_distribution_plot",
            "description": "对指定的数据列绘制分布图并保存",
            "parameters": {
                "type": "object",
                "properties": {
                    "column_name": {"type": "string", "description": "列名，例如 'Survived' 或 'Sex'"},
                    "title": {"type": "string", "description": "图表标题"}
                },
                "required": ["column_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_sklearn_classification",
            "description": "使用随机森林对指定目标进行分类预测",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_column": {"type": "string", "description": "预测目标列"},
                    "feature_columns": {"type": "array", "items": {"type": "string"}, "description": "特征列列表"}
                },
                "required": ["target_column", "feature_columns"]
            }
        }
    }
]