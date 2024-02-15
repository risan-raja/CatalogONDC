import torch

class SpareseResults(torch.nn.Module):
    def __init__(self):
        super().__init__()
    

    def forward(self,mlm_logits):
        with torch.no_grad():
            batch_size = mlm_logits.size(0)
            mlm_nz = mlm_logits.nonzero()
            vec_indices = torch.vstack((mlm_nz[:,0], mlm_nz[:,1]))
            vec_values = mlm_logits[mlm_nz[:,0], mlm_nz[:,1]]
            del mlm_logits
            results = torch.zeros((batch_size, 2,  512))
            for row in range(batch_size):
                indices = torch.zeros(512)  # type: ignore
                values = torch.zeros(512)  # type: ignore
                mask = vec_indices[0].eq(row)
                row_indices = torch.masked_select(vec_indices[1], mask)
                indices[:row_indices.shape[0]] = row_indices
                row_values = torch.masked_select(vec_values, mask)
                values[:row_values.shape[0]] = row_values
                result = torch.vstack((indices, values))
                results[row] = result
            return results


srm = SpareseResults()
srm = torch.jit.script(srm)

srm.save("splade_models/sparse_results.pt") # type: ignore