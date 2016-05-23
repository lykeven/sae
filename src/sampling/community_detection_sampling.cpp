#include "community_detection_sampling.h"
#include "../basic/community_detection.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdio>
#include <queue>
#include <set>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace sae::io;

Community_detection_sampling::Community_detection_sampling(MappedGraph *graph)
    :Solver(graph){
}

Community_detection_sampling::~Community_detection_sampling() {
}


MappedGraph * select_sub_graph(MappedGraph *graph,int& num,map<vid_t,vid_t>& map_to,int K)
{
    GraphBuilder<int> sub_graph_builder;
    map<vid_t,vid_t> is_part;
    int n=graph->VertexCount(),m=graph->EdgeCount();
    auto viter = graph->Vertices();

    for(int i=0;i<num;i++)
    {

        vid_t sample_v=i;
        map_to[i]=sample_v;
        is_part[sample_v]=i;
        sub_graph_builder.AddVertex(i,0);
    }
    for(int i=0;i<num;i++)
    {
        vid_t origin_v=map_to[i];
        viter->MoveTo(origin_v);
        for(auto eiter = viter->OutEdges(); eiter->Alive(); eiter->Next())
            if(is_part.find(eiter->TargetId())!=is_part.end())
                sub_graph_builder.AddEdge(i,is_part[eiter->TargetId()],0);
    }
    sub_graph_builder.Save("./data/temp");
    MappedGraph *sub_graph = MappedGraph::Open("./data/temp");
    return sub_graph;
}

vector<vid_t> allocate_vertex_with_shortest_path(MappedGraph *graph,vector<vid_t> community,int sample_num)
{
        int n=graph->VertexCount();
        vector<vid_t> communityChange(n,1);
        for(int i=0;i<sample_num;i++)
            communityChange[i]=community[i];
        for(int i=sample_num;i<n;i++)
        {
            queue < int >  search_queue;
            vector< int > dis(n,-1);
            dis[i] = 0;
            search_queue.push(i);
            auto viter = graph->Vertices();
            while(!search_queue.empty())
            {
                int v=search_queue.front();
                search_queue.pop();
                viter->MoveTo(v);
                for(auto eiter = viter->OutEdges(); eiter->Alive(); eiter->Next())
                {
                    int w = eiter->TargetId();
                    if(dis[w] < 0)
                    {
                        search_queue.push(w);
                        dis[w] = dis[v] + 1;
                    }
                }
            }
            int min_value=10000,min_index=0;
            for(int j=0;j<sample_num;j++)
                if(dis[j]>0&&dis[j]<min_value)
                {
                    min_value=dis[j];
                    min_index=j;
                }
            communityChange[i]=community[min_index];
        }
        return communityChange;
}

vector<vid_t> allocate_vertex_with_shortest_path2(MappedGraph *graph,vector<vid_t> community,int sample_num,int K,map<vid_t,vid_t> map_to)
{
        int n=graph->VertexCount();
        vector<vid_t> communityChange(n,1);
        vector<vid_t> is_part(n,0);
        for(int i=0;i<sample_num;i++)
        {
                communityChange[map_to[i]]=community[i];
                is_part[map_to[i]]=1;
        }

        vector <int> t(K,-1);
        vector <vector<int>> dis(n,t);
        for(int k=0;k<K;k++)
        {
            queue < int >  search_queue;
            for(int i=0;i<sample_num;i++)
            {
                    vid_t origin_v=map_to[i];
                    if (community[origin_v]==k+1)
                    {
                        dis[origin_v][k] = 0;
                        search_queue.push(origin_v);
                    }
            }
            auto viter = graph->Vertices();
            while(!search_queue.empty())
            {
                int v=search_queue.front();
                search_queue.pop();
                viter->MoveTo(v);
                for(auto eiter = viter->OutEdges(); eiter->Alive(); eiter->Next())
                {
                    int w = eiter->TargetId();
                    if(dis[w][k] < 0)
                    {
                        search_queue.push(w);
                        dis[w][k] = dis[v][k] + 1;
                    }
                }
            }
        }
        for(int j=0;j<n;j++)
        {
            if(is_part[j]==1) continue;
            for(int i=0;i<dis[j].size();i++) if(dis[j][i]==-1) dis[j][i]=1000;
            int min_index=distance(dis[j].begin(), min_element(dis[j].begin(), dis[j].end()));
            communityChange[j]=min_index+1;
        }
        return communityChange;

}

pair<vector<vid_t>,double> allocate_vertex_with_label_propagation(MappedGraph *graph,vector<vid_t> community,int sample_num,int K)
{
        srand(time(NULL));
        unsigned int  n=graph->VertexCount();
        vector<vector<vid_t>> neighbor(n);
        vector<vid_t> C(n,1);
        vector<vid_t> temp(n);
        vector<vid_t> max_community=C;
        double max_modularity=0;
        vid_t max_community_count=K;
        auto viter = graph->Vertices();
        for (unsigned int i=0;i<n;i++)
        {
                temp[i]=i;
                if(i<sample_num)
                    C[i]=community[i];
                else{
                    C[i]=rand()%max_community_count+1;
                    viter->MoveTo(i);
                    for(auto eiter = viter->OutEdges(); eiter->Alive(); eiter->Next())
                        neighbor[i].push_back(eiter->TargetId());
                }
        }
        bool is_success=false;
        while(!is_success)
        {
                is_success=true;
                vector<vid_t> rand_seq=temp;
                for (int j =n-1;j>=0;j--)
                {
                        int x = rand() % (n - 1);
                        unsigned tem =rand_seq[j];
                        rand_seq[j]=rand_seq[x];
                        rand_seq[x]=tem;
                }
                for(unsigned int j=0;j<n;j++)
                {
                        unsigned v=rand_seq[j];
                        if (v<sample_num) continue;
                        vector <vid_t> vertex_neigh=neighbor[v];
                        for (unsigned int k=0;k<vertex_neigh.size();k++)
                            vertex_neigh[k]=C[vertex_neigh[k]];
                        sort(vertex_neigh.begin(),vertex_neigh.end());
                        int max_count=1,new_count=1,max_vertex=vertex_neigh[0],new_vertex=vertex_neigh[0];
                        for (unsigned int k=1;k<vertex_neigh.size();k++)
                        {
                                if (vertex_neigh[k]==new_vertex) new_count++;
                                if(vertex_neigh[k]!=max_vertex||k==vertex_neigh.size()-1)
                                {
                                    if(new_count>max_count)
                                    {
                                            max_count=new_count;
                                            max_vertex=new_vertex;
                                    }
                                    new_vertex=vertex_neigh[k];
                                    new_count=1;
                                }
                        }
                        if(max_count==1) max_vertex=vertex_neigh[rand()%vertex_neigh.size()];
                        if (C[v]!=max_vertex)
                        {
                                C[v]=max_vertex;
                                is_success=false;
                        }
                }
            Community_detection cc(graph);
            double modularity=cc.compute_modularity(graph,C,max_community_count);
            if (modularity>max_modularity) {
                max_modularity=modularity;max_community=C;
            }
        }
            return make_pair(max_community,max_modularity);
}


pair<vector<vid_t>,double> fixed_community(MappedGraph *graph,vector<vid_t> community,double mod,int K)
{
    int n=graph->VertexCount(),m=graph->EdgeCount() ,weight=0;
    vector<vid_t> nodes(n),community_new(n);
    vector<vector<vid_t>>edges(m),node_edges(n), region(K);
    vector<int> eii(K),ai(K),ki(n);
    auto viter = graph->Vertices();
    //initialize parameter
    for(int i=0;i<n;i++)
    {
        nodes[i]=i;
        viter->MoveTo(i);
        community_new[i]=community[i]-1;
        region[community[i]-1].push_back(i);
        ki[i]=viter->OutEdgeCount();
        ai[community_new[i]]+=ki[i];
        weight+=ki[i];
        for(auto eiter = viter->OutEdges(); eiter->Alive(); eiter->Next())
        {
            int j=eiter->GlobalId();
            edges[j]=vector<vid_t> {static_cast<unsigned long long>(i),eiter->TargetId(),1};
            node_edges[i].push_back(j);
            if(community[i]==community[eiter->TargetId()]) eii[community_new[i]]+=1;
        }
    }
        while(1){
            bool is_change=false;
            //remove i and update para
            for(int i=0;i<n;i++)
            {
                //if(region[C[i]].size()!=1) continue;
                region[community_new[i]].erase(remove(region[community_new[i]].begin(),
                 region[community_new[i]].end(), i), region[community_new[i]].end());
                vector<vid_t> neighbor_list=node_edges[i];
                int share_weight=0,c1=community_new[i];
                for(int j=0;j<neighbor_list.size();j++)
                {
                    int v=neighbor_list[j];
                    int s=edges[v][0],e=edges[v][1],w=edges[v][2];
                    if(c1==community_new[e])    share_weight+=w;
                }
                eii[c1]-=2*share_weight;
                ai[c1]-=ki[i];
                int best_gain=0,best_share_weight=0,best_community=c1,share_weight2=0;
                //find a neighbor which have largest gain
                for(int j=0;j<neighbor_list.size();j++)
                {
                    int v=neighbor_list[j];
                    if(v==i) continue;
                    int s=edges[v][0],e=edges[v][1],w=edges[v][2],c2=community_new[e];
                    share_weight2=0;
                    for(int j=0;j<neighbor_list.size();j++)
                    {
                        int v=neighbor_list[j];
                        int s=edges[v][0],e=edges[v][1],w=edges[v][2];
                        if(c2==community_new[e])    share_weight2+=w;
                    }
                    //compute q gain and find the largest one
                    double gain=2*share_weight2-ai[c2]*ki[i]*1.0/weight;
                    if(gain>best_gain){
                        best_gain=gain;
                        best_share_weight=share_weight2;
                        best_community=c2;
                    }
                }
                //add i to neighbor's community and update para
                if(best_gain<=0) {best_community=c1;best_share_weight=share_weight;}
                region[best_community].push_back(i);
                community_new[i]=best_community;
                eii[best_community]+=2*best_share_weight;
                ai[best_community]+=ki[i];
                if(c1!=best_community)  is_change=true;
            }
            if(!is_change)
                break;
        }
    //compute modularity
    double q=0;
    int k=1;
    for(int i=0;i<K;i++)
        if(region[i].size()>=0)
            q+=eii[i]*1.0/weight-(ai[i]*1.0/weight)*(ai[i]*1.0/weight);
     map<vid_t,vid_t> map_seq;
    for(int i=0;i<region.size();i++)
    {
        if(region[i].size()>0){
            for(int j=0;j<region[i].size();j++)
                map_seq[region[i][j]]=k;
            k++;
        }
    }
    for(int i=0;i<n;i++)
        community_new[i]=map_seq[i];
    return make_pair(community_new,q);
}


void recurPermutation(vector<vid_t> c,vector<vid_t> c2,double * max_overlap,vector<int> arr, int n, int i)
{
    if(i==n-1) {
        int overlapping =0;
        for (int i=0;i<c.size();i++)
            if(arr[c[i]-1]==c2[i])
                overlapping++;
        double overlap=overlapping*1.0/c.size();
        if (overlap>*max_overlap)
            *max_overlap=overlap;
    }
    for(int j=i; j<n; j++) {
        swap(arr[i], arr[j]);
        recurPermutation(c,c2,max_overlap,arr, n, i+1);
        swap(arr[i], arr[j]);
    }
}

void  Community_detection_sampling::test_community_sampling(MappedGraph *graph,int min_rate,int max_rate,int min_k,int max_k,int sub_task)
{
    Community_detection cd(graph);
    cout<<"\tRun community detection sampling algorithm"<<endl;
    double overlap[(max_rate-min_rate)/10+1][max_k-min_k+1];
    double mod1[(max_rate-min_rate)/10+1][max_k-min_k+1];
    double mod2[(max_rate-min_rate)/10+1][max_k-min_k+1];
    double t1[(max_rate-min_rate)/10+1][max_k-min_k+1];
    double t2[(max_rate-min_rate)/10+1][max_k-min_k+1];
    double rate[(max_rate-min_rate)/10+1][max_k-min_k+1];
    for (int K=min_k;K<=max_k;K++)
    {
        time_t start_time = clock();
        pair<vector<vid_t>,double> ans=cd.solve(K,sub_task);
        time_t end_time = clock();

        cout<<endl<<"\tmodularity is "<<ans.second<<endl;
        for (int p=min_rate;p<=max_rate;p+=10)
        {
            cout<<"\n\n\sampling rate:"<<p<<"%"<<"\tcommunity number:"<<K<<endl<<endl;
            int n=graph->VertexCount();
            time_t start_time2 = clock();
            pair<vector<vid_t>,double> ans2=solve(p/100.0,K,sub_task);
            cout<<endl<<"\tmodularity is "<<ans2.second<<endl;
            time_t end_time2 = clock();
             t1[int(p-min_rate)/10][K-min_k]=(end_time- start_time+ 0.0) / CLOCKS_PER_SEC ;
            t2[int(p-min_rate)/10][K-min_k]=(end_time2- start_time2 + 0.0) / CLOCKS_PER_SEC ;
            rate[int(p-min_rate)/10][K-min_k]=t1[int(p-min_rate)/10][K-min_k]/t2[int(p-min_rate)/10][K-min_k];

            double  overlapping=0;
            double *point_overlap=&overlapping;
            vector<int> a(K);
            for (int i=0;i<K;i++)
                a[i]=i+1;
            recurPermutation(ans.first,ans2.first,point_overlap,a,K,0);
            overlap[int(p-min_rate)/10][K-min_k]=*point_overlap;
            mod1[int(p-min_rate)/10][K-min_k]=ans.second;
            mod2[int(p-min_rate)/10][K-min_k]=ans2.second;
        }
    }
    FILE* fout = fopen("output/community_detection_sampling/table" ,"w");
    fprintf(fout, "rate\t");
    for (int K=min_k;K<=max_k;K++)
        fprintf(fout, "mod1\tmod2\tt1\tt2\toverlap\t");
    fprintf(fout, "\n");
    for (int p=min_rate;p<=max_rate;p+=10)
        {
            fprintf(fout, "%d%%\t",p);
            for (int K=min_k;K<=max_k;K++)
            fprintf(fout, "%.4f\t%.4f\t%.2f\t%.2f\t%.2f\t",mod1[int(p-min_rate)/10][K-min_k],mod2[int(p-min_rate)/10][K-min_k],
            t1[int(p-min_rate)/10][K-min_k],t2[int(p-min_rate)/10][K-min_k],overlap[int(p-min_rate)/10][K-min_k]);
             fprintf(fout, "\n");
        }
}

pair<vector<vid_t>,double>  Community_detection_sampling::solve(double p,int K,int sub_task)
{
    int n=graph->VertexCount();
    int sample_num=n*p;
    map<vid_t,vid_t> map_to;
    MappedGraph * sub_graph= select_sub_graph(graph,sample_num,map_to,K);

    Community_detection cd(sub_graph);

    pair<vector<vid_t>,double> ans=cd.solve(K,sub_task);
    int k=*max_element(ans.first.begin(),ans.first.end());
    cout<<endl<<endl<<"\tmodularity is "<<ans.second<<"\tK is "<<k<<endl;
    //vector<vid_t> final_community=allocate_vertex_with_shortest_path(graph,ans.first,sample_num);
    vector<vid_t> final_community=allocate_vertex_with_shortest_path2(graph,ans.first,sample_num,k,map_to);
    double modularity=cd.compute_modularity(graph,final_community,k);
    cout << endl<<endl<<"\tafter assign vertex " <<"\tmodularity is "<<modularity<<endl;
    return make_pair(final_community,modularity);

//     ans=fixed_community(graph,final_community,modularity,k);
//     k=*max_element(ans.first.begin(),ans.first.end());
//     cout<<endl<<endl<<"\tafter fixed community " << "\tmodularity is "<<ans.second<<"\tK is "<<k<<endl;
//      return ans;

	//pair<vector<vid_t>,double> final_community=allocate_vertex_with_label_propagation(graph,ans.first,sample_num,k);
	//return final_community;
}
