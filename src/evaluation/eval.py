import json
url = "http://web.megatechai.com:33612/process_facts"
headers = {
  'Content-Type': 'application/json'
}
import requests

ground_truth = json.load(open('../../data/dataset.json', 'r'))

start = 0
end = 300
date = 1205
mode = "key_answer" # key_answer, key_middle, bertscore

method = ["plan_solve", "plan_execute",'react']
model_names = ["glm-4", "glm-4-plus", 'llama3.1-instruct', 'qwen-max', 'claude-3-5-sonnet-20241022','gpt-3.5-turbo',  "gpt-4o-mini-2024-07-18",'gpt-4o-2024-08-06']

scores = {
    
}

for model_name in model_names:
    scores[model_name] = []
    for p in method:
        test_datas = open(f'../output_{date}/{p}_{date}_{model_name}.json', 'r').readlines()

        average_score = 0

        if mode == "key_answer":

            for idx, data in enumerate(test_datas[start:end]):
                # print(data)
                data = json.loads(data)

                res = data['res']
                key = ground_truth[start + idx]['key']
                # print(res, key)
                total = len(key)

                score = 0

                for k in key:
                    if res.find(k) != -1:
                        score += 1
                score /= total
                scores[model_name].append(score)
                average_score += score

            print(format(average_score / len(test_datas[start:end]), ".4f"))
        
        elif mode == "key_middle":

            for idx, data in enumerate(test_datas[start:end]):
                data = json.loads(data)

                res = data['summary']
                
                key_answer = ground_truth[start + idx]['key']
                key_middle = ground_truth[start + idx]['key_middle']
                key = []
                key.extend(key_answer)
                key.extend(key_middle)

                total = len(key)

                score = 0

                for k in key:
                    if res.find(k) != -1:
                        score += 1
                score /= total
                average_score += score

            print(format(average_score / len(test_datas[start:end]), ".4f"))

        elif mode == "bertscore":
            answers = []
            labels = []
            for idx, data in enumerate(test_datas[start:end]):
                data = json.loads(data)
                res = data['res']
                answers.append(res)
                label = ground_truth[start + idx]['answer']
                labels.append(label)
                

            payload = json.dumps({"answer_facts": answers, "label_facts": labels})
            response = requests.request("POST", url, headers=headers, data=payload)
            F1 = sum(json.loads(response.text)['F1']) / len(json.loads(response.text)['F1'])

            print(format(F1, ".4f"))



