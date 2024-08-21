import json


run_ikat24 = json.load(open("run_GPT4o_QR_ikat24.json"))
ans_dict = {}
for line in open("queries_ANS_GPT4o_ikat24.tsv"):
    conv_turn_id, ans = line.split("\t")
    ans_dict[conv_turn_id] = ans


clean_run = {"run_name": "sample_run",
             "run_type": "automatic",
             "eval_response": True,
             "turns": []}


for q, ranking in run_ikat24.items():
    clean_ranking = {"turn_id": q,
                     "responses":[
                         {
                             "rank": 1,
                             "text": ans_dict[q],
                             "ptkb_provenance": [],
                             "passage_provenance": []
                         }
                     ]}
    for i, (d, s) in enumerate(ranking.items()):
        clean_retrieved_doc = {"id": d,
                               "score": s,
                               "used": i<=5}
        clean_ranking["responses"][0]["passage_provenance"].append(clean_retrieved_doc)

    clean_run["turns"].append(clean_ranking)

json.dump(clean_run,open("clean_run_GPT4o_QR_ikat24.json", "w"),indent=4)