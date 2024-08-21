from openai import OpenAI
import json
from tqdm import tqdm

API_key = "[openai-token]"
client = OpenAI(
  api_key=API_key
)


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
        # context_q = ""
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


prompt_string = """# Instruction:
I will give you a conversation between a user and a system. Also, I will give you some background about the user. You should rewrite the last question of the user into a self-contained query.

# Background knowledge:
{ptkb}
# Context:
{ctx}
# Please rewrite the following user question:
{user_utterance}
# Re-written query: 
"""


with open("queries_QR_GPT4o_ikat24.tsv", "w") as tsv_queries_raw_4o:

    for turn_id, ctx_user in tqdm(data.items()):

        ptkb=ctx_user[0]
        ctx = ctx_user[1]
        user_utterance=ctx_user[2]

        response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt_string.format(ptkb=ptkb, ctx=ctx, user_utterance=user_utterance)}
        ],
        )

        response = response.choices[0].message.content
        tsv_queries_raw_4o.write(turn_id+"\t"+response+"\n")
