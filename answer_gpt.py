from openai import OpenAI
import json
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

API_key = "[openai-token]"
client = OpenAI(
  api_key=API_key
)

index_path = "[path-to-lucene-index]"
searcher = LuceneSearcher(index_path)
run = json.load(open("run_GPT4o_QR_ikat24.json"))

data={}
with open("[path-to-topics]","r") as tc:
    q = dict()
    rw = dict()
    a = dict()

    topics = json.load(tc)
    for topic in topics:
        topic_number = str(topic["number"])
        q[topic_number] = dict()
        rw[topic_number] = dict()
        a[topic_number] = dict()
        context = ""
        ptkb_raw = topic["ptkb"]
        ptkb=""
        for k,v in ptkb_raw.items():
            ptkb+=str(k)+ ": " + str(v) + ", "
        ptkb = ptkb[:-2] # remove additional ", " from the end
        for index, turn in enumerate(topic['turns']):
            turn_id = str(turn['turn_id'])
            topic_turn_id = str(topic_number) + "_" + str(turn_id)

            user_utterance=turn['utterance']
            data[topic_turn_id]=(ptkb,context,user_utterance)

            context=context+"\nuser: "+turn['utterance']
            context = context.strip()

            context=context+"\nsystem: "+turn["response"]
            context = context.strip()


prompt_string = """# Doc1: {doc1}
# Doc2: {doc2}
# Doc3: {doc3}
# Doc4: {doc4}
# Doc5: {doc5}
# I will give you a conversation between a user and a system. Also, I will give you some background information about the user. You should answer the last utterance of the user by providing a summary of the relevant parts of the given documents. Please remember that your answer shouldn't be more than 200 words.
# Background information about the user:
{ptkb}
# Conversation: 
{ctx}
# User query:
{user_utterance}
"""


with open("queries_ANS_GPT4o_ikat24.tsv", "w") as tsv_queries_raw_4o:
    for turn_id, ctx_user in tqdm(data.items()):

        doc_id_ranking_top5 = [doc_id for doc_id, score in run[turn_id].items()][:5]
        ranking_top5_raw = [searcher.doc(doc_id) for doc_id in doc_id_ranking_top5]
        ranking_top5 = [json.loads(doc.raw())['contents'].strip().replace("\n", " ") for doc in ranking_top5_raw]

        ptkb=ctx_user[0]
        ctx = ctx_user[1]
        user_utterance=ctx_user[2]

        response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt_string.format(ptkb=ptkb, ctx=ctx, user_utterance=user_utterance, doc1=ranking_top5[0], doc2=ranking_top5[1],doc3=ranking_top5[2],doc4=ranking_top5[3],doc5=ranking_top5[4])}
        ],
        )

        response = response.choices[0].message.content.replace("\n", " ")
        tsv_queries_raw_4o.write(turn_id+"\t"+response+"\n")
