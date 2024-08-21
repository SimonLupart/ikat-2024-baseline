from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
import json
from sentence_transformers import CrossEncoder


index_path = "[path-to-lucene-index]"
searcher = LuceneSearcher(index_path)

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')

def run_bm25_model_one_query(query_text, num_passages_returned_by_bm25):

    bm_25_outout = searcher.search(query_text, k=num_passages_returned_by_bm25)
    passages = []
    for hit in bm_25_outout:
        passages.append(hit.docid)
    return passages

def rerank_results_by_bm25_one_query(passages, query_text):

    passage_text_mapping = searcher.batch_doc(passages, threads=128)
    re_ranked_results = []
    query_passage_pairs = [ [query_text, json.loads(passage_text_mapping[passage_id].raw())['contents']] for passage_id in passages]
    scores = reranker.predict(query_passage_pairs, batch_size=256, show_progress_bar=False)   

    for pass_id, score in zip(passages, scores):
        re_ranked_results.append([pass_id, score])     
    re_ranked_results_sorted = sorted(re_ranked_results, key=lambda x: x[1], reverse=True)
    return re_ranked_results_sorted, passage_text_mapping

def get_top_n_passages_returned_by_model(re_ranked_results_sorted, n, passage_text_mapping):

    top_passages = []
    for i in range(0, n):
        pass_id, score = re_ranked_results_sorted[i]
        tmp = {'text':json.loads(passage_text_mapping[pass_id].raw())['contents'],
                             'id': pass_id,
                             'score': score,
                             'rank': i+1}
        top_passages.append(tmp)
    return top_passages


def run_ranking_pipeline_one_query(query_text, n):

    passages = run_bm25_model_one_query(query_text, n)
    re_ranked_results_sorted, passage_text_mapping = rerank_results_by_bm25_one_query(passages, query_text)
    top_passages = get_top_n_passages_returned_by_model(re_ranked_results_sorted, n, passage_text_mapping)

    return top_passages


# *********************************
ranking={}
with open("queries_QR_GPT4o_ikat24.tsv") as rw_file:
  for line in tqdm(rw_file):
      i, query_text = line.strip().split("\t")

      ranking_q={}
      num_returned_passages = 1000
      out_ranking = run_ranking_pipeline_one_query(query_text, num_returned_passages)

      for doc in out_ranking:
        ranking_q[doc["id"]]=float(doc["score"])

      ranking[i]=ranking_q

json.dump(ranking, open("run_GPT4o_QR_ikat24.json", "w"))