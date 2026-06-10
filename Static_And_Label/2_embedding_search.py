import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer



# 1. 读取数据
file_path = 'data/工作写作标注结果.xlsx'
# 读取 Query 和 评分列
df = pd.read_excel(file_path, usecols=['随机Q', '返修'])

# 确保没有空值
df = df.dropna(subset=['随机Q'])
queries = df['随机Q'].tolist()
scores = df['返修'].tolist()

# 2. 生成向量集合 (使用 embedding3-0.6b 模型)
# 注意：请确保模型名称正确。如果是 HuggingFace 上的模型，请填写完整路径
# 假设该模型兼容 SentenceTransformer 格式
model_name = '/root/paddlejob/workspace/env_run/output/zacharychu/afs_data/model/Qwen3-1.7B' 
model = SentenceTransformer(model_name, device="cuda")
# 显式使用 AutoModel (基础模型) 而不是 AutoModelForCausalLM (带头的模型)
# model = AutoModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
# model.cuda()
print("正在生成向量，请稍候...")
query_embeddings = model.encode(queries, normalize_embeddings=True, show_progress_bar=True)
print("生成向量，完成...")

# 3. 构建 Faiss 向量库 A
dimension = query_embeddings.shape[1]  # 向量维度
index = faiss.IndexFlatIP(dimension)   # 使用 L2 距离（欧氏距离）衡量相似度
index.add(np.array(query_embeddings).astype('float32')) # 将向量添加到索引中

# 4. 执行检索
q = "什么的成语"
top_k = 5  # 召回前 5 个最相似的结果

# 将查询语句转化为向量


def process_embedding(q, model, queries, scores, index, top_k = 10):
    search_vector = model.encode([q], normalize_embeddings=True).astype('float32')
    distances, indices = index.search(search_vector, top_k)
    result = []
    bad_nums = []
    for i in range(top_k):
        idx = indices[0][i]
        if distances[0][i] >  0.99:
            result.append([queries[idx], "%.4f" % distances[0][i], scores[idx]])
            if scores[idx] == 0:
                bad_nums.append(1)
    
    return result

def process_bad_number(result):
    bad_nums = []
    for res in result:
        if res[2] == 0:
            bad_nums.append(1)
    return len(bad_nums) / (len(result)+0.5)

# 在 Faiss 中进行搜索
# distances: 相似度距离, indices: 匹配到的原始数据索引
# distances, indices = index.search(search_vector, top_k)

print("no use！")
dataframe = pd.read_excel("output/验收数据/文创线上数据_清洗数据.xlsx")
dataframe["embedding_sim"] = dataframe["Query"].apply(lambda query: process_embedding(query, model, queries, scores, index, top_k = 10))
dataframe.head()
dataframe["vote"] = dataframe["embedding_sim"].apply(lambda res: process_bad_number(res))
dataframe.to_excel("output/验收数据/文创线上数据_清洗数据_vote.xlsx", index=None)


