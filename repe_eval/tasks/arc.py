from datasets import load_dataset
import numpy as np
from .utils import shuffle_all_train_choices

def arc_dataset(config, ntrain=25, seed=1):
    template_str = "Consider the correctness of the answer to the following question:\nQuestion: {question}\nAnswer: {answer}.\nThe probability the answer being correct is "

    def clean_answer_s(s):
        return s[:-1] if s[-1] == "." else s
    
    def format_samples(df, idx):
        prompts = []

        question = df['question'][idx]
        prompt_choices_text =  df['choices'][idx]['text'].tolist() # 选项文本的列表
        prompt_choices_label = df['choices'][idx]['label'].tolist() # 选项label的列表 ['A','B','C','D']
        
        answer_choice =  df['answerKey'][idx] # 真实label的string
        true_answer_s =  prompt_choices_text[prompt_choices_label.index(answer_choice)] # 真实choice的string

        prompts.append(template_str.format(question=question, answer=clean_answer_s(true_answer_s))) # answer不需要.号，因为模板中带了
        
        for false_answer_s, i in [(a,i) for i,a in enumerate(prompt_choices_text) if a != true_answer_s]:
            prompts.append(template_str.format(question=question, answer=clean_answer_s(false_answer_s)))    

        return prompts, [1] + [0] * (len(prompt_choices_label) - 1) # 1代表正确，0代表错误，第一个是正确的

    def _keep_4_options_row(e):
        return len(e['choices']['label']) == 4

    def samples(df):
        prompts, labels = [], []
        for i in range(df.shape[0]):
            answer_prompts, label =  format_samples(df, i)
            prompts.append(answer_prompts)
            labels.append(label)
        return prompts, labels

    dataset = load_dataset("ai2_arc", config)
    train_df = dataset['train'].filter(_keep_4_options_row).shuffle(seed=seed).to_pandas()
    test_df = dataset['test'].to_pandas()
    val_df = dataset['validation'].to_pandas()

    train_data, train_labels = samples(train_df) # 全都是带prompt的
    test_data, test_labels = samples(test_df)
    val_data, val_labels = samples(val_df)


    train_data, train_labels =  train_data[:ntrain], train_labels[:ntrain] # 选择多少个构建PCA
    train_data, train_labels = shuffle_all_train_choices(train_data, train_labels, seed) # 打乱选项顺序

    train_data =  np.concatenate(train_data).tolist()
    test_data =  np.concatenate(test_data).tolist()
    val_data = np.concatenate(val_data).tolist()

    return {
            "train": {"data": train_data, "labels": train_labels}, 
            "test": {"data": test_data, "labels": test_labels}, 
            "val": {"data": val_data, "labels": val_labels}
            }