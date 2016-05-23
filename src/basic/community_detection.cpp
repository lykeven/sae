#include "community_detection.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdio>
#include <queue>
#include <map>
#include <set>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace sae::io;

Community_detection::Community_detection(MappedGraph *graph)
    :Solver(graph){
}

Community_detection::~Community_detection() {
}


vector<pair<double, int> > compute_betweenness(MappedGraph *graph,vector< bool> is_deleted)
{
    vector<pair<double, int> > edge_betweenness;
    unsigned int n = graph->VertexCount();
     for(auto eiter = graph->Edges(); eiter->Alive(); eiter->Next())
            edge_betweenness.push_back(make_pair(0,eiter->GlobalId()));
    for(unsigned int s = 0; s < n; s++)
    {
        vector< vector<eid_t> > pre_vertex(n);
        vector< int >  search_queue;
        vector< int > point_stack;
        vector< int > path_count(n,0);
        vector< int > dis(n,-1);
        vector<double> part_value(n,0);
        dis[s] = 0;
        path_count[s] = 1;
        search_queue.push_back(s);
        auto viter = graph->Vertices();
        unsigned int head=-1;
        while(head+1<search_queue.size())
        {
            int v=search_queue[++head];
            point_stack.push_back(v);
            viter->MoveTo(v);
            for(auto eiter = viter->OutEdges(); eiter->Alive(); eiter->Next())
            {
                if(is_deleted[eiter->GlobalId()]) continue;
                int w = eiter->TargetId();
                if(dis[w] < 0)
                {
                    search_queue.push_back(w);
                    dis[w] = dis[v] + 1;
                }
                if(dis[w] == dis[v] + 1)
                {
                    path_count[w] += path_count[v];
                    pre_vertex[w].push_back(v);
                }
            }
        }

        for(int i = point_stack.size()-1; i >= 0; i--)
        {
            unsigned int w = point_stack[i];
            if(w==s) continue;
            for(unsigned int j = 0;j < pre_vertex[w].size(); j++)
            {
                unsigned int wp=pre_vertex[w][j];
                part_value[wp] += (double)path_count[wp] / path_count[w] * (1 + part_value[w]);
                if(wp==s||w==s) continue;
                auto vex = graph->Vertices();
                vex->MoveTo(wp);
                for(auto eiter = vex->OutEdges(); eiter->Alive(); eiter->Next())
                {
                    if(is_deleted[eiter->GlobalId()]) continue;
                    if(eiter->TargetId()==w)
                    {
                        edge_betweenness[eiter->GlobalId()].first +=(double)path_count[wp] / path_count[w] * part_value[w];
                        break;
                    }
                }
            }
        }
    }
    sort(edge_betweenness.begin(), edge_betweenness.end());
    reverse(edge_betweenness.begin(), edge_betweenness.end());
    return edge_betweenness;
}

pair<vector<vid_t>,int >  split_community(MappedGraph *graph,vector< bool> is_deleted)
{
    unsigned int n=graph->VertexCount(),k=1;
    vector <vid_t> community(n,0);
    for (unsigned int i=0;i<n;i++)
    {
        if(community[i]==0)
        {
            queue < int >  search_queue;
            search_queue.push(i);
            auto viter = graph->Vertices();
            while(!search_queue.empty())
            {
                    int v=search_queue.front();
                    search_queue.pop();
                    community[v]=k;
                    viter->MoveTo(v);
                    for(auto eiter = viter->OutEdges(); eiter->Alive(); eiter->Next())
                    {
                        if(is_deleted[eiter->GlobalId()]) continue;
                        int w = eiter->TargetId();
                        if(community[w]== 0)
                        {
                            search_queue.push(w);
                            community[w]=k;
                        }
                    }
            }
        k++;
        }
    }
    return make_pair(community,k-1);
}


double  Community_detection::compute_modularity(MappedGraph *graph, vector<eid_t> community,int k)
{
    int m = graph->EdgeCount();
    vector< vector<eid_t> > community_set(k);
    for (unsigned int i=0;i<community.size();i++)
            community_set[community[i]-1].push_back(i);
    double ans=0,eii=0,ai=0;
    auto viter = graph->Vertices();
    for (int i=0;i<k;i++)
    {
        double eii_temp=0,ai_temp=0;
        for(unsigned int j=0;j<community_set[i].size();j++)
        {
                viter->MoveTo(community_set[i][j]);
                for(auto eiter = viter->OutEdges(); eiter->Alive(); eiter->Next())
                {
                        if (community[eiter->TargetId()]==i+1)
                            eii_temp+=1;
                        ai_temp+=1;
                }
        }
        eii=eii_temp/m;
        ai=ai_temp/m;
        ans+=eii-ai*ai;
    }
    return ans;
}


pair<vector<vid_t>,double>   Community_detection::run_Girvan_NewMan(MappedGraph *graph,int K)
{
    int m=graph->EdgeCount();
    int n= graph->VertexCount();
    vector< vid_t> community(n,1);
    vector< bool> is_deleted(m,false);
    vector<double> modularity(m,0);
    int k=1,j=0;
    for (;j<m;j++)
    {
        vector<pair<double, int> >edge_betweenness =compute_betweenness(graph,is_deleted);
        for (unsigned int c=0;c<edge_betweenness.size();c++)
            if(edge_betweenness[c].first==edge_betweenness[0].first)
            {
                int first =edge_betweenness[c].second;
                int second=edge_betweenness[c].second%2==0?edge_betweenness[c].second+1:edge_betweenness[c].second-1;
                is_deleted[first]=true;
                is_deleted[second]=true;
            }

        modularity[j]=modularity[j-1];
        if(edge_betweenness[0].first==0)  break;
        pair<vector<vid_t>,int > split_com=split_community(graph,is_deleted);
        community=split_com.first;
        k=split_com.second;
        modularity[j]=compute_modularity(graph,community,k);
        if(k>=K) break;
    }
    return make_pair(community,modularity[j]);
}

pair<vector<vid_t>,double>   Community_detection::run_label_propagation(MappedGraph *graph)
{
        srand(time(NULL));
        unsigned int  n=graph->VertexCount();
        vector<vector<vid_t>> neighbor(n);
        vector<vid_t> C(n);
        vector<vid_t> temp(n);
        vector<vid_t> max_community(n);
        double max_modularity=0;
        for (unsigned int i=0;i<n;i++)
        {
                C[i]=i;temp[i]=i;
                auto viter = graph->Vertices();
                viter->MoveTo(i);
                for(auto eiter = viter->OutEdges(); eiter->Alive(); eiter->Next())
                    neighbor[i].push_back(eiter->TargetId());
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
            set<vid_t> community;
            set<vid_t>::reverse_iterator rit;
            for(unsigned int i=0;i<n;i++)
                community.insert(C[i]);
            map<vid_t,vid_t> map_seq;
            int seq=1,community_count=community.size();
            for (rit = community.rbegin(); rit != community.rend(); rit++)
                    map_seq[*rit]=seq++;
            vector <vid_t> final_community=C;
            for(unsigned int i=0;i<C.size();i++)
                    final_community[i]=map_seq[C[i]];
            double modularity=compute_modularity(graph,final_community,community_count);
            if (modularity>max_modularity) {
                max_modularity=modularity;max_community=final_community;
            }
        }
            return make_pair(max_community,max_modularity);
}

vector<int> BFS(MappedGraph *graph,int s,int threshold)
{
        int n=graph->VertexCount();
        queue < int >  search_queue;
        vector< int > dis(n,-1);
        dis[s] = 0;
        search_queue.push(s);
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
                else if(dis[w]>threshold)
                return dis;
            }
        }
        return dis;
}


vector<vid_t> get_community_core(MappedGraph *graph,int K,int  threshold_length,double threshold_rate)
{
    int n=graph->VertexCount();
    vector<pair<int, int> > indegree;
    auto viter = graph->Vertices();
    for (int i=0;i<n;i++)
    {
        viter->MoveTo(i);
        indegree.push_back(make_pair(viter->OutEdgeCount(),i));
    }
    sort(indegree.begin(), indegree.end());
    reverse(indegree.begin(), indegree.end());

    int k=1;
    vector <vid_t>com_core_set;
    vector <vid_t>community(n,0);
    com_core_set.push_back(0);
    for(int i=0;i<n;i++)
    {
        if(k>=K) break;
        int v=indegree[i].second;
        if(community[v]==0)
        {
            vector<int > dis=BFS(graph,i,threshold_length);
            vector<vid_t> temp=community;
            int overlap=0,total=0;
            for(int j=0;j<dis.size();j++)
                if(dis[j]==threshold_length)
                {
                    total++;
                    if(temp[j]==0){
                        overlap++;
                        temp[j]=k;
                    }
                }
            if(overlap*1.0/total>=threshold_rate)
            {
                community=temp;
                com_core_set.push_back(v);
                k++;
            }
        }
    }
    return com_core_set;
}

vector<vid_t> allocate_vertex(MappedGraph *graph,vector<vid_t> com_core_set)
{
        int K=com_core_set.size();
        int n=graph->VertexCount();
        vector<vid_t> communityChange(n,0);
        vector <int> t(K,-1);
        vector <vector<int>> dis(n,t);
        for(int k=0;k<K;k++)
        {
            queue < int >  search_queue;
            int s=com_core_set[k];
            dis[s][k] = 0;
            search_queue.push(s);
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
            for(int i=0;i<dis[j].size();i++) if(dis[j][i]==-1) dis[j][i]=1000;
            int min_index=distance(dis[j].begin(), min_element(dis[j].begin(), dis[j].end()));
            communityChange[j]=min_index+1;
        }
        return communityChange;
}

pair<vector<vid_t>,double> Community_detection::run_k_community_core(MappedGraph *graph,int K)
{
    int D=0,iteration=10,n=graph->VertexCount();
    for(int i=0;i<iteration;i++)
    {
        int s=rand() % (n - 1);
        vector<int> dis=BFS(graph,s,1000);
        int new_s=distance(dis.begin(), max_element(dis.begin(), dis.end()));
        dis=BFS(graph,new_s,1000);
        int d=*max_element(dis.begin(), dis.end());
        if(d>D)
            D=d;
    }
    //cout<<"\tD :"<<D<<endl<<endl;
    int threshold_length=D/K;
    vector<double>rate={0.75,0.9};
    vector<int>length={2,3,4};
    vector<vid_t> community,best_community;
    double modularity,best_modularity=0;
    for(int i=0;i<rate.size();i++)
    for(int j=0;j<length.size();j++)
    {
        vector<vid_t> com_core_set=get_community_core(graph,K,length[j],rate[i]);
        community= allocate_vertex(graph, com_core_set);
        modularity=compute_modularity(graph,community,com_core_set.size());
        //cout<<"\tlength: "<<length[j]<<"\trate: "<<rate[i]<<endl;
        //cout<<"\tcommunity core number"<<com_core_set.size()<<"\tmodularity is "<<modularity<<endl;
        if(modularity>best_modularity){
            best_community=community;
            best_modularity=modularity;
        }
    }
    return make_pair(best_community,best_modularity);
}

pair<vector<vid_t>,double> Community_detection::run_louvain_method(MappedGraph *graph)
{
    int n=graph->VertexCount(),m=graph->EdgeCount();
    vector<vid_t> nodes(n),nodes_new,community(n);
    vector<vector<vid_t>> edges(m),edges_new;
    vector<vector<vid_t>>node_edges(n),node_edges_new;
    auto viter = graph->Vertices();
    //initialize parameter
    for(int i=0;i<n;i++)
    {
        nodes[i]=i;
        viter->MoveTo(i);
        community[i]=i;
        for(auto eiter = viter->OutEdges(); eiter->Alive(); eiter->Next())
        {
            int j=eiter->GlobalId();
            edges[j]=vector<vid_t> {static_cast<unsigned long long>(i),eiter->TargetId(),1};
            node_edges[i].push_back(j);
        }
    }

    double best_q=-1;
    while(1){
            //initialize parameter begin each loop for phase 1 and 2
            int nn=nodes.size(),mm=edges.size(),weight=0;;
            vector<int> eii(nn),ai(nn),ki(nn),wi(nn);
            vector<vector<vid_t>> region(nn);
            vector<vid_t> C(nn);
            for(int i=0;i<nn;i++)
            {
                C[i]=i;
                region[i]=vector<vid_t>{static_cast<unsigned long long>(i)};
                vector<vid_t> neighbor_list=node_edges[i];
                for(int j=0;j<neighbor_list.size();j++)
                {
                    int v=neighbor_list[j];
                    int s=edges[v][0],e=edges[v][1],w=edges[v][2];
                    if(s==e){
                        wi[i]+=w;eii[i]+=w;
                    }
                    ai[i]+=w;ki[i]+=w;weight+=w;
                }
            }
        while(1){
            //phase 1
            bool is_change=false;
            //remove i and update para
            for(int i=0;i<nn;i++)
            {
                //if(region[C[i]].size()!=1) continue;
                region[C[i]].erase(remove(region[C[i]].begin(), region[C[i]].end(), i), region[C[i]].end());
                vector<vid_t> neighbor_list=node_edges[i];
                int share_weight=0,c1=C[i];
                for(int j=0;j<neighbor_list.size();j++)
                {
                    int v=neighbor_list[j];
                    int s=edges[v][0],e=edges[v][1],w=edges[v][2];
                    if(s==e) continue;
                    if(c1==C[e])    share_weight+=w;
                }
                eii[c1]-=(2*(share_weight)+wi[i]);
                ai[c1]-=ki[i];
                int best_gain=0,best_share_weight=0,best_community=c1,share_weight2=0;
                //find a neighbor which have largest gain
                for(int j=0;j<neighbor_list.size();j++)
                {
                    int v=neighbor_list[j];
                    if(v==i) continue;
                    int s=edges[v][0],e=edges[v][1],w=edges[v][2],c2=C[e];
                    share_weight2=0;
                    for(int j=0;j<neighbor_list.size();j++)
                    {
                        int v=neighbor_list[j];
                        int s=edges[v][0],e=edges[v][1],w=edges[v][2];
                        if(s==e) continue;
                        if(c2==C[e])    share_weight2+=w;
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
                region[best_community].push_back(i);
                C[i]=best_community;
                eii[best_community]+=(2*(best_share_weight)+wi[i]);
                ai[best_community]+=ki[i];
                if(c1!=best_community)  is_change=true;
            }
            if(!is_change)
                break;
        }
        //compute modularity
        double q=0;
        for(int i=0;i<nn;i++)
            if(region[i].size()>=0)
                q+=eii[i]*1.0/weight-(ai[i]*1.0/weight)*(ai[i]*1.0/weight);
        if(q<=best_q)   break;
        if(q>best_q)    best_q=q;
        //phase 2
        int k=0,p=0;
        map<vid_t,vid_t> map_seq;
        for(int i=0;i<nn;i++)
        {
            if(region[i].size()>0){
                nodes_new.push_back(k);
                for(int j=0;j<region[i].size();j++)
                    map_seq[region[i][j]]=k;
                k++;
            }
        }
        for(int i=0;i<n;i++)
            community[i]=map_seq[community[i]];
        vector<vector<vid_t>> temp(k);
        vector<vid_t> self_lop(k);
        node_edges_new=temp;
        for(int i=0;i<mm;i++)
        {
            int s=edges[i][0],e=edges[i][1],w=edges[i][2];
            int c1=map_seq[C[s]],c2=map_seq[C[e]];
            if(c1!=c2)
            {
                edges_new.push_back(vector<vid_t>{static_cast<unsigned long long>(c1),static_cast<unsigned long long>(c2),static_cast<unsigned long long>(w)});
                node_edges_new[c1].push_back(p);
                p++;
            }
            else    self_lop[c1]+=w;
        }
        for(int i=0;i<self_lop.size();i++)
        {
            if(self_lop[i]!=0){
                edges_new.push_back(vector<vid_t>{static_cast<unsigned long long>(i),static_cast<unsigned long long>(i),self_lop[i]});
                node_edges_new[i].push_back(p);
                p++;
            }
        }
        nodes=nodes_new;
        edges=edges_new;
        node_edges=node_edges_new;
        nodes_new.clear();edges_new.clear();node_edges_new.clear();
    }
    for(int i=0;i<n;i++)
        community[i]=community[i]+1;
    return make_pair(community,best_q);
}

pair<vector<vid_t>,double> Community_detection::solve(int K,int sub_task)
{
    pair<vector<vid_t>,double> community;
    switch(sub_task){
    case 1:community=run_Girvan_NewMan(graph,K);break;
    case 2:community=run_label_propagation(graph);break;
    case 3:community=run_louvain_method(graph);break;
    case 4:community=run_k_community_core(graph,K);break;
    default:community=run_Girvan_NewMan(graph,K);break;
    }
    return community;
}
