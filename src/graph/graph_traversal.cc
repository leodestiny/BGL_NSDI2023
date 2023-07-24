/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/traversal.cc
 * \brief Graph traversal implementation
 */
#include <dgl/graph_traversal.h>
#include <dgl/packed_func_ext.h>
#include "../c_api_common.h"
#include <queue>
#include <cstdlib>

using namespace dgl::runtime;

namespace dgl {
namespace traverse {


void BFSTraverseNodesWithMask(const aten::CSRMatrix& csr,
		int64_t source,
		NDArray mask,
		bool* visited, int64_t* bfs_seq){
  const char * mask_data = static_cast<char*>(mask->data); 

  const int64_t *indptr_data = static_cast<int64_t *>(csr.indptr->data);
  const int64_t *indices_data = static_cast<int64_t *>(csr.indices->data);
  const int64_t num_nodes = csr.num_rows;
  memset(visited, 0, sizeof(bool) * num_nodes);
  //std::vector<bool> visited(num_nodes);
  std::queue<int64_t> Q;
  //std::vector<int64_t> bfs_seq;
  int64_t seq_cnt = 0;
 
  
  visited[source] = true;
  Q.push(source);
  if(mask_data[source]){
    bfs_seq[seq_cnt++] = source;
  }

  // traverse entire graph
  while(!Q.empty()){
    int64_t u = Q.front();
    Q.pop();
    for(auto idx = indptr_data[u]; idx < indptr_data[u+1]; ++idx){
      auto v = indices_data[idx];
      if (!visited[v]) {
        visited[v] = true;
        Q.push(v);
        if(mask_data[v]){
	  bfs_seq[seq_cnt++] = v;
	}
      }
    }
  }

  //ensure all node are traversed
  int64_t s = source;
  for(int64_t i = 0; i < num_nodes; ++i){
    s++;
    if(s == num_nodes)
      s = 0;
    if(visited[s])
      continue;

    visited[s] = true;
    Q.push(s);
    if(mask_data[s]){
      bfs_seq[seq_cnt++] = s;
    }
    
    // traverse entire graph
    while(!Q.empty()){
      int64_t u = Q.front();
      Q.pop();
      for(auto idx = indptr_data[u]; idx < indptr_data[u+1]; ++idx){
        auto v = indices_data[idx];
        if (!visited[v]) {
          visited[v] = true;
          Q.push(v);
          if(mask_data[v]){
            bfs_seq[seq_cnt++] = v;
          }
        }
      }
    }
  }
}

DGL_REGISTER_GLOBAL("traversal._CAPI_DGLMultipleBFSNodesWithMask")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef g = args[0];
    const IdArray src = args[1];
    bool reversed = args[2];
    aten::CSRMatrix csr;
    if (reversed) {
      csr = g.sptr()->GetCSCMatrix(0);
    } else {
      csr = g.sptr()->GetCSRMatrix(0);
    }
    NDArray mask = args[3];
    int64_t mask_nonzero_count = args[4];
    int64_t bfs_pass = args[5];
    const int64_t *src_data = static_cast<int64_t*>(src->data);

    NDArray ret = NDArray::Empty({mask_nonzero_count*bfs_pass}, {kDLInt, 64, 1}, {kDLCPU, 0});
    int64_t* ret_data = static_cast<int64_t*>(ret->data);

    const int64_t num_nodes = csr.num_rows;
    bool* visited = new bool[num_nodes];
    int64_t* bfs_seq = new int64_t[mask_nonzero_count];
    int64_t offset = mask_nonzero_count / bfs_pass;
    for(int64_t i = 0; i < bfs_pass; ++i){
      BFSTraverseNodesWithMask(csr, src_data[i], mask, visited, bfs_seq);
      for(int64_t j = 0; j < mask_nonzero_count; ++j)
        ret_data[j*bfs_pass+i] = bfs_seq[(j+i*offset)%mask_nonzero_count];
    }
    delete[] visited;
    delete[] bfs_seq;
    *rv = ret;
  });


DGL_REGISTER_GLOBAL("traversal._CAPI_DGLBFSNodes_v2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef g = args[0];
    const IdArray src = args[1];
    bool reversed = args[2];
    aten::CSRMatrix csr;
    if (reversed) {
      csr = g.sptr()->GetCSCMatrix(0);
    } else {
      csr = g.sptr()->GetCSRMatrix(0);
    }
    const auto& front = aten::BFSNodesFrontiers(csr, src);
    *rv = ConvertNDArrayVectorToPackedFunc({front.ids, front.sections});
  });

DGL_REGISTER_GLOBAL("traversal._CAPI_DGLBFSEdges_v2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef g = args[0];
    const IdArray src = args[1];
    bool reversed = args[2];
    aten::CSRMatrix csr;
    if (reversed) {
      csr = g.sptr()->GetCSCMatrix(0);
    } else {
      csr = g.sptr()->GetCSRMatrix(0);
    }

    const auto& front = aten::BFSEdgesFrontiers(csr, src);
    *rv = ConvertNDArrayVectorToPackedFunc({front.ids, front.sections});
  });

DGL_REGISTER_GLOBAL("traversal._CAPI_DGLTopologicalNodes_v2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef g = args[0];
    bool reversed = args[1];
    aten::CSRMatrix csr;
    if (reversed) {
      csr = g.sptr()->GetCSCMatrix(0);
    } else {
      csr = g.sptr()->GetCSRMatrix(0);
    }

    const auto& front = aten::TopologicalNodesFrontiers(csr);
    *rv = ConvertNDArrayVectorToPackedFunc({front.ids, front.sections});
  });


DGL_REGISTER_GLOBAL("traversal._CAPI_DGLDFSEdges_v2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef g = args[0];
    const IdArray source = args[1];
    const bool reversed = args[2];
    CHECK(aten::IsValidIdArray(source)) << "Invalid source node id array.";
    aten::CSRMatrix csr;
    if (reversed) {
      csr = g.sptr()->GetCSCMatrix(0);
    } else {
      csr = g.sptr()->GetCSRMatrix(0);
    }
    const auto& front = aten::DGLDFSEdges(csr, source);
    *rv = ConvertNDArrayVectorToPackedFunc({front.ids, front.sections});
  });

DGL_REGISTER_GLOBAL("traversal._CAPI_DGLDFSLabeledEdges_v2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef g = args[0];
    const IdArray source = args[1];
    const bool reversed = args[2];
    const bool has_reverse_edge = args[3];
    const bool has_nontree_edge = args[4];
    const bool return_labels = args[5];
    aten::CSRMatrix csr;
    if (reversed) {
      csr = g.sptr()->GetCSCMatrix(0);
    } else {
      csr = g.sptr()->GetCSRMatrix(0);
    }

    const auto& front = aten::DGLDFSLabeledEdges(csr,
                                                 source,
                                                 has_reverse_edge,
                                                 has_nontree_edge,
                                                 return_labels);

    if (return_labels) {
      *rv = ConvertNDArrayVectorToPackedFunc({front.ids, front.tags, front.sections});
    } else {
      *rv = ConvertNDArrayVectorToPackedFunc({front.ids, front.sections});
    }
  });

}  // namespace traverse
}  // namespace dgl
