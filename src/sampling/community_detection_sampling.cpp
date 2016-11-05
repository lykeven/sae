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

MappedGraph * FP(MappedGraph *graph,int& num,map<vid_t,vid_t>& map_to,int K,map<vid_t,vid_t>& is_part)
{
    for(int i=0;i<num;i++)
    {
        is_part[i]=i;
        map_to[i]=i;
    }
}


MappedGraph * RNS(MappedGraph *graph,int& num,map<vid_t,vid_t>& map_to,int K,map<vid_t,vid_t>& is_part)
{
    int n=graph->VertexCount();
    vector<vid_t> temp(n);
    for (unsigned int i=0;i<n;i++)  temp[i]=i;
    vector<vid_t> rand_seq=temp;
    for (int j =n-1;j>=0;j--)
    {
        int x = rand() % (n - 1);
        unsigned tem =rand_seq[j];
        rand_seq[j]=rand_seq[x];
        rand_seq[x]=tem;
    }
    for(int i=0;i<num;i++)
    {
        vid_t v= rand_seq[i];
        is_part[v]=i;
        map_to[i]=v;
    }
}

MappedGraph * RNI(MappedGraph *graph,int& num,map<vid_t,vid_t>& map_to,int K,map<vid_t,vid_t>& is_part)
{
    int n=graph->VertexCount(),m=graph->EdgeCount();
    vector<pair<int, int> > indegree;
    auto viter = graph->Vertices();
    for (int i=0;i<n;i++)
    {
        viter->MoveTo(i);
        indegree.push_back(make_pair(viter->OutEdgeCount(),i));
    }
    sort(indegree.begin(), indegree.end());
    reverse(indegree.begin(), indegree.end());
    int max_indegree=indegree[0].first+1,max_v=0;
    vector<vector<vid_t>> indegree_dis(max_indegree);
    vector<vid_t> rand_list(num);
    for (int i=0;i<n;i++)
        indegree_dis[indegree[i].first].push_back(indegree[i].second);
    for(int i=0;i<max_indegree;i++)
    {
        int m=indegree_dis[i].size();
        if(m==0) continue;
        int k=m*num/n+1;
        for(int j=0;j<k;j++)
        {
            int rand_index=rand()%k;
            if(find(rand_list.begin(),rand_list.end(),indegree_dis[i][rand_index])!=rand_list.end()){j--;continue;}
            if(max_v>=num) break;
            rand_list[max_v++]=indegree_dis[i][rand_index];
        }
    }
    for(int i=0;i<num;i++)
    {
        vid_t v =rand_list[i];
        is_part[v]=i;
        map_to[i]=v;
    }
}

MappedGraph * RJ(MappedGraph *graph,int& num,map<vid_t,vid_t>& map_to,int K,map<vid_t,vid_t>& is_part)
{
    int n=graph->VertexCount(),sample_num=0;
    double jump_p=0.15;
    vector<vector<vid_t>> outdegree(n);
    map<vid_t,vid_t>::iterator it;
    auto viter = graph->Vertices();
    for(int i=0;i<n;i++)
    {
        viter->MoveTo(i);
        for(auto eiter = viter->OutEdges(); eiter->Alive(); eiter->Next())
            outdegree[i].push_back(eiter->TargetId());
    }
    int s=rand() % n;
    is_part[s]=sample_num;
    map_to[sample_num]=s;
    sample_num++;
    while(1)
    {
        if(sample_num>=num) break;
        viter->MoveTo(s);
        bool is_jump=rand()%100<15?true:false;
        if(is_jump)
        {
            s=rand() % n;
            it = is_part.find(s);
            if (it == is_part.end())
            {
                is_part[s]=sample_num;
                map_to[sample_num]=s;
                sample_num++;
            }
        }
        else
        {
            int m=outdegree[s].size();
            if(m==0) continue;
            int  rand_num=rand() % m ;
            s=outdegree[s][rand_num];
            it = is_part.find(s);
            if (it == is_part.end())
            {
                is_part[s]=sample_num;
                map_to[sample_num]=s;
                sample_num++;
            }
        }
    }
}

MappedGraph * MAXI(MappedGraph *graph,int& num,map<vid_t,vid_t>& map_to,int K,map<vid_t,vid_t>& is_part)
{
    int n=graph->VertexCount(),m=graph->EdgeCount();
    auto viter = graph->Vertices();
    vector<pair<int, int> > indegree;
    for (int i=0;i<n;i++)
    {
        viter->MoveTo(i);
        indegree.push_back(make_pair(viter->OutEdgeCount(),i));
    }
    sort(indegree.begin(), indegree.end());
    reverse(indegree.begin(), indegree.end());
    for(int i=0;i<num;i++)
    {
        vid_t v =indegree[i].second;
        is_part[v]=i;
        map_to[i]=v;
    }
}

MappedGraph * CC(MappedGraph *graph,int& num,map<vid_t,vid_t>& map_to,int K,map<vid_t,vid_t>& is_part)
{
    int n=graph->VertexCount(),m=graph->EdgeCount();
    vector<int > indegree(n),is_exist(n,0);
    vector<pair<int, int> > indegree_sort;
    vector<vector<vid_t>> node_target(n);
    auto viter = graph->Vertices();
    int T=num/K;
    for (int i=0;i<n;i++)
    {
        viter->MoveTo(i);
        vid_t num=viter->OutEdgeCount();
        indegree[i]=num;
         indegree_sort.push_back(make_pair(num,i));
        vector<vid_t> temp(num);
        node_target[i]=temp;
        int k=0;
        for(auto eiter = viter->OutEdges(); eiter->Alive(); eiter->Next())
            node_target[i][k++]=eiter->TargetId();
    }
    sort(indegree_sort.begin(), indegree_sort.end());
    reverse(indegree_sort.begin(), indegree_sort.end());
    set<vid_t> kernel_set;
    set<vid_t>::reverse_iterator rit;
     int k=0;
    for(int i=0;i<K;i++)
    {
        set<vid_t> kernel_new;
        vid_t rand_v=indegree_sort[k].second;
        while(is_exist[rand_v])
        {
            rand_v=indegree_sort[++k].second;
            if(k>=n) break;
        }
        if(k>=n) break;
        kernel_set.insert(rand_v);
        kernel_new.insert(rand_v);
        is_exist[rand_v]=i+1;
        for(int j=1;j<T;j++)
        {
            int max_con=0,max_indegree=0,max_index=0;
            for (rit = kernel_new.rbegin(); rit != kernel_new.rend(); rit++)
            {
                for(int c=0;c<node_target[*rit].size();c++)
                {
                    int neighbor_v=node_target[*rit][c],con=0;
                    if(is_exist[neighbor_v]) continue;
                    for(int b=0;b<node_target[neighbor_v].size();b++)
                        if(is_exist[node_target[neighbor_v][b]]) con++;
                    if(con>=max_con)
                    {
                        max_con=con;
                        if(max_indegree<indegree[neighbor_v])
                        {
                            max_indegree=indegree[neighbor_v];
                            max_index=neighbor_v;
                        }
                    }
                }
            }
            is_exist[max_index]=i+1;
            kernel_set.insert(max_index);
            kernel_new.insert(max_index);
        }
    }
    num=0;
    for(int i=0;i<n;i++)
    {
        if(is_exist[i])
        {
            vid_t sample_v=num;
            is_part[i]=sample_v;
            map_to[sample_v]=i;
            num++;
        }
    }
}

MappedGraph * select_sub_graph(MappedGraph *graph,int& num,map<vid_t,vid_t>& map_to,int K,int sm)
{
    GraphBuilder<int> sub_graph_builder;
    map<vid_t,vid_t> is_part;
    switch(sm)
    {
        case 0: FP(graph, num,map_to, K,is_part);break;
        case 1: RNS(graph, num,map_to, K,is_part);break;
        case 2: RNI(graph, num,map_to, K,is_part);break;
        case 3: RJ(graph, num,map_to, K,is_part);break;
        case 4: MAXI(graph, num,map_to, K,is_part);break;
        case 5: CC(graph, num,map_to, K,is_part);break;
        default:FP(graph, num,map_to, K,is_part);break;
    }
    int n=graph->VertexCount(),m=graph->EdgeCount();
    auto viter = graph->Vertices();
    for(int i=0;i<num;i++)
            sub_graph_builder.AddVertex(i,0);
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

vector<vid_t> allocate_vertex_with_shortest_path(MappedGraph *graph,vector<vid_t> &community,int sample_num,int K,map<vid_t,vid_t> &map_to)
{
        int n=graph->VertexCount();
        vector<vid_t> communityChange(n,1),is_part(n,0);;
        for(int i=0;i<sample_num;i++)
        {
            communityChange[map_to[i]]=community[i];
            is_part[map_to[i]]=1;
        }
        for(int i=0;i<n;i++)
        {
            if(is_part[i]==1) continue;
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
            int min_value=10000,min_index=rand()%sample_num;
            for(int j=0;j<sample_num;j++)
                if(dis[map_to[j]]>0&&dis[map_to[j]]<min_value)
                {
                    min_value=dis[map_to[j]];
                    min_index=j;
                }
            communityChange[i]=community[min_index];
        }
        return communityChange;
}

vector<vid_t> allocate_vertex_with_shortest_path2(MappedGraph *graph,vector<vid_t> &community,int sample_num,int K,map<vid_t,vid_t> &map_to)
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

vector<vid_t>allocate_vertex_with_label_propagation(MappedGraph *graph,vector<vid_t> &community,int sample_num,int K,map<vid_t,vid_t> &map_to)
{
        srand(time(NULL));
        unsigned int  n=graph->VertexCount();
        vector<vector<vid_t>> neighbor(n);
        vector<vid_t> C(n,1),temp(n),max_community(n,1),is_part(n,0);
        for(int i=0;i<sample_num;i++)
        {
            C[map_to[i]]=community[i];
            is_part[map_to[i]]=1;
        }
        double max_modularity=0;
        vid_t max_community_count=K;
        auto viter = graph->Vertices();
        for (unsigned int i=0;i<n;i++)
        {
            temp[i]=i;
            if(is_part[i]==1)   continue;
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
                        if(is_part[v]==1) continue;
                        vector <vid_t> label=neighbor[v];
                        for (unsigned int k=0;k<label.size();k++)
                            label[k]=C[label[k]];
                        sort(label.begin(),label.end());
                        int max_count=1,max_v=label[0],new_count=1,new_v=label[0];
                        vector<int>max_list;
                        for(int i=1;i<label.size();i++)
                        {
                            if(label[i]==new_v) new_count++;
                            if(label[i]!=new_v||i==label.size()-1)
                            {
                                if(new_count==max_count)
                                    max_list.push_back(new_v);
                                if(new_count>max_count)
                                {
                                    max_list.clear();
                                    max_list.push_back(new_v);
                                    max_count=new_count;
                                    max_v=new_v;
                                }
                                if(i==(label.size()-1)&&max_count==1) max_list.push_back(label[i]);
                                new_count=1;
                                new_v=label[i];
                            }
                        }
                        if(max_list.size()==0) max_v=label[0];
                        else  max_v=max_list[rand()%max_list.size()];
                        if (C[v]!=max_v)
                        {
                                C[v]=max_v;
                                is_success=false;
                        }
                }
        }
        return C;
}


pair<vector<vid_t>,double> fixed_community(MappedGraph *graph,vector<vid_t> &community,double mod,int K)
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


void recurPermutation(vector<vid_t> &c,vector<vid_t> &c2,double * max_overlap,vector<int> &arr, int n, int i)
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

void  Community_detection_sampling::test_community_sampling(MappedGraph *graph,int sub_task,int sm,int ex)
{
    Community_detection cd(graph);
    double P[]={0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9};
    int K[]={2,4,6};
    int n1=sizeof(P) / sizeof(P[0]), n2=sizeof(K) / sizeof(K[0]);
    cout<<"\tRun community detection sampling algorithm"<<endl;
    double overlap[n1][n2],mod[n2],mod1[n1][n2],mod2[n1][n2],t1[n1][n2],t2[n1][n2],rate[n1][n2];
    for (int i=0;i<n2;i++)
    {
        time_t start_time = clock();
        pair<vector<vid_t>,double> ans=cd.solve(K[i],sub_task);
        time_t end_time = clock();
        mod[i]=ans.second;
        cout<<endl<<"\tmodularity is "<<ans.second<<endl;
        for (int j=0;j<n1;j++)
        {
            cout<<"\n\n\sampling rate:"<<P[j]<<"\tcommunity number:"<<K[i]<<endl<<endl;
            time_t start_time2 = clock();
            int n=graph->VertexCount(), sample_num=n*P[j];
            map<vid_t,vid_t> map_to;
            MappedGraph * sub_graph= select_sub_graph(graph,sample_num,map_to,K[i],sm);

            Community_detection cd2(sub_graph);
            pair<vector<vid_t>,double> ans2=cd2.solve(K[i],sub_task);
            int k=*max_element(ans2.first.begin(),ans2.first.end());
            cout<<endl<<endl<<"\tmodularity is "<<ans2.second<<"\tK is "<<k<<endl;
            mod1[j][i]=ans2.second;
            vector<vid_t> community(n);
            switch(ex)
            {
                case 1:community=allocate_vertex_with_shortest_path(graph,ans.first,sample_num,k,map_to);break;
                case 2:community=allocate_vertex_with_shortest_path2(graph,ans.first,sample_num,k,map_to);break;
                case 3:community=allocate_vertex_with_label_propagation(graph,ans.first,sample_num,k,map_to);break;
                default:community=allocate_vertex_with_shortest_path2(graph,ans.first,sample_num,k,map_to);break;
            }
            double modularity=cd.compute_modularity(graph,community,k);
            cout << endl<<endl<<"\tafter assign vertex " <<"\tmodularity is "<<modularity<<endl<<endl;;
            pair<vector<vid_t>,double> ans3=solve(P[j],K[i],sub_task,sm,ex);
            cout<<endl<<"\tmodularity is "<<ans3.second<<endl;
            time_t end_time2 = clock();
             t1[j][i]=(end_time- start_time+ 0.0) / CLOCKS_PER_SEC ;
            t2[j][i]=(end_time2- start_time2 + 0.0) / CLOCKS_PER_SEC ;
            rate[j][i]=t1[j][i]/t2[j][i];

            double  overlapping=0;
            double *point_overlap=&overlapping;
            vector<int> a(K[i]);
            for (int k=0;k<K[i];k++)
                a[k]=k+1;
            recurPermutation(ans.first,ans3.first,point_overlap,a,K[i],0);
            overlap[j][i]=*point_overlap;
            mod2[j][i]=ans3.second;
        }
    }
    FILE* fout = fopen("output/table" ,"w");
    fprintf(fout, "rate\t");
    for (int i=0;i<n2;i++)
        fprintf(fout, "mod\tmod1\tmod2\tt1\tt2\toverlap\t");
    fprintf(fout, "\n");
    for (int i=0;i<n1;i++)
        {
            fprintf(fout, "%.1f%%\t",P[i]*100);
            for (int j=0;j<n2;j++)
            fprintf(fout, "%.4f\t%.4f\t%.4f\t%.2f\t%.2f\t%.2f\t",mod[j],mod1[i][j],mod2[i][j],t1[i][j],t2[i][j],overlap[i][j]);
             fprintf(fout, "\n");
        }
}

pair<vector<vid_t>,double>  Community_detection_sampling::solve(double p,int K,int sub_task,int sm,int ex)
{
    int n=graph->VertexCount(), sample_num=n*p;
    map<vid_t,vid_t> map_to;
    MappedGraph * sub_graph= select_sub_graph(graph,sample_num,map_to,K,sm);

    Community_detection cd(sub_graph);
    pair<vector<vid_t>,double> ans=cd.solve(K,sub_task);
    int k=*max_element(ans.first.begin(),ans.first.end());
    cout<<endl<<endl<<"\tmodularity is "<<ans.second<<"\tK is "<<k<<endl;

     vector<vid_t> community(n);
     switch(ex)
     {
        case 1:community=allocate_vertex_with_shortest_path(graph,ans.first,sample_num,k,map_to);break;
        case 2:community=allocate_vertex_with_shortest_path2(graph,ans.first,sample_num,k,map_to);break;
        case 3:community=allocate_vertex_with_label_propagation(graph,ans.first,sample_num,k,map_to);break;
        default:community=allocate_vertex_with_shortest_path2(graph,ans.first,sample_num,k,map_to);break;
     }
    double modularity=cd.compute_modularity(graph,community,k);
    cout << endl<<endl<<"\tafter assign vertex " <<"\tmodularity is "<<modularity<<endl<<endl;;
    return make_pair(community,modularity);

//     ans=fixed_community(graph,final_community,modularity,k);
//     k=*max_element(ans.first.begin(),ans.first.end());
//     cout<<endl<<endl<<"\tafter fixed community " << "\tmodularity is "<<ans.second<<"\tK is "<<k<<endl;
//      return ans;
}
