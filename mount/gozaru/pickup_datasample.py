import pandas as pd 
file = "./databricks-dolly-15k-ja-gozaru/databricks-dolly-15k-ja-gozaru.json"
trainPath = './input.txt'

# JSONファイルを読み込む 
df = pd.read_json(file) 
df_train = df[df['category'] == 'open_qa'] 
retBuffer = list()
for index, dfs in df_train.iterrows():
    inputText ="[INST]"+dfs.instruction+'[/INST]'+dfs.output.replace('\n','')
    if len(inputText) < 64:         
        retBuffer.append(inputText)

# 保存 
with open(trainPath, 'w') as file:
    file.write('\n'.join(retBuffer))
