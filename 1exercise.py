import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 示例数据
data = {
    "mpg": [21, 22, 23, 24, 25],
    "price": [10000, 12000, 14000, 16000, 18000]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 添加常数项（截距）
df = sm.add_constant(df)
print(df.head())  # 检查是否成功添加 'const' 列

# 运行 OLS 线性回归
ols_model = sm.OLS(df["price"].astype(float), df[["const", "mpg"]].astype(float)).fit()
# 输出回归结果
print(ols_model.summary())

# 将回归结果保存到文本文件
with open("regression_results.txt", "w") as f:
    f.write(ols_model.summary().as_text())

print("回归结果已保存到 regression_results.txt")

# 可视化回归结果
plt.scatter(df["mpg"], df["price"], label="Actual Data")
plt.plot(df["mpg"], ols_model.predict(df[["const", "mpg"]]), color="red", label="Regression Line")
plt.xlabel("Miles per Gallon (mpg)")
plt.ylabel("Price ($)")
plt.title("Regression Analysis: mpg vs price")
plt.legend()
plt.grid()
plt.show()



# 取用户输入的新的 mpg 值
new_mpg = float(input("输入一个新的 mpg 值以预测 price: "))

# 构造新的数据点 DataFrame
new_data = pd.DataFrame({"const": [1], "mpg": [new_mpg]})
predicted_price = ols_model.predict(new_data)[0]
print(f"预测的价格为: ${predicted_price:.2f}")

