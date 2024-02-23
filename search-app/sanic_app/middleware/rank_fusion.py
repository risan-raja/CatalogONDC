def convert_to_runs(combined_results):
    runs = []
    run_dict = {}
    run_dict["q_1"] = {
        k: combined_results["dense_vector_search"][k]["score"]
        for k in combined_results["dense_vector_search"]
    }
    runs.append(run_dict)
    run_dict = {}
    run_dict["q_1"] = {
        k: combined_results["sparse_vector_search"][k]["score"]
        for k in combined_results["sparse_vector_search"]
    }
    runs.append(run_dict)
    return runs


def fuse_rank(combined_json):
    if combined_json:
        print("Fusing ranks")
        if combined_json["rerank"]:
            print("Reranking")
            combined_results = combined_json["combined_results"]
            last = None
            if combined_results["structured_search"]:
                last = combined_results["structured_search"].copy()
            del combined_results["structured_search"]
            runs = convert_to_runs(combined_results)
            r1 = runs[0]["q_1"]
            r2 = runs[1]["q_1"]
            min_r1 = min(r1.values())
            max_r1 = max(r1.values())
            min_r2 = min(r2.values())
            max_r2 = max(r2.values())
            r1 = {k: (v - min_r1) / (max_r1 - min_r1) for k, v in r1.items()}
            r2 = {k: (v - min_r2) / (max_r2 - min_r2) for k, v in r2.items()}
            # Combine the runs
            combined_runs = {}
            alpha = 0.7
            common_keys = set(r1.keys()).intersection(set(r2.keys()))
            for k in common_keys:
                combined_runs[k] = alpha * r1[k] + (1 - alpha) * r2[k]
            for k in set(r1.keys()).difference(common_keys):
                combined_runs[k] = r1[k] * alpha
            for k in set(r2.keys()).difference(common_keys):
                combined_runs[k] = r2[k] * (1 - alpha)
            combined_runs = {
                k: v
                for k, v in sorted(
                    combined_runs.items(), key=lambda item: item[1], reverse=True
                )
            }
            # combine the results using id
            results = {}
            for rec in combined_runs:
                if rec in combined_results["dense_vector_search"]:
                    results[rec] = combined_results["dense_vector_search"][rec]
                    results[rec]["score"] = combined_runs[rec]
                else:
                    results[rec] = combined_results["sparse_vector_search"][rec]
                    results[rec]["score"] = combined_runs[rec]
            if last:
                for rec in last:
                    if rec not in results:
                        results[rec] = last[rec]
                        results[rec]["score"] = 1e-5
            del combined_json["combined_results"]
            combined_json["results"] = results
            # print(combined_json.raw_body)
            return combined_json
        else:
            return combined_json
