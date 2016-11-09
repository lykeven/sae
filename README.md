## SAE: Social Analytic Engine - MarkI
### Usage
/bin/sae –i STRING:INPUT [-o STRING:OUTPUT] [-t STRING:TASK] [TASK REPEVANT PARAMETERS]

#### Parameter Description
-i/--input: declare the direction of input graph

-o/--output: declare the direction of output files

-t/--task: declare the task, with the options as

md: make and save sae graph from file

mt: make and save sae graph from file for tencent weibo data

im: run influence maximization;

[-k INT:K]: the size of seed set;

[-w STRING:WEIGHT]: set the weights of edges, with options as

rand: assign weights randomly within (0, 1);

deg: for a directed edge (u, v), let its weight to be 1/[out-degree-of-u]);

const: assign a constant weight to all edges;

A parameter is followed to declare the constant value:

-c FLOAT:CONSTANT-WEIGHT

pr: run PageRank;

sp: shortest path;

dd: demonstrate the degree distribution;

tr: count the number of triangles;

cd: community detection;

[-k INT:K]: the number of community;

[-r INT:run]: choose a algorithm for community detection;

cs: community detection with sampling:

[-k INT:K]: the number of community;

[-p FLOAT:probability]: sample probability for community detection;

ne: network embedding; 

[-r INT:run]: choose a algorithm for network embedding;

te: test network embedding; 

[-q INT:label file]: choose a label file for network embedding on multi-label classfication task;

sr: run SimRank ;

[-r INT:run]: choose a algorithm for SimRank;

[-c FLOAT:constant]: weaken factor for SimRank, which between 0~1,default value is 0.8;

[-s INT:start]: the querying node, which should be exits in the dataset;

[-q STRING:query file]: the querying file, which contain query nodes.Each line has a node number;

[-k INT:K]: the number of Top K;

psm: propensity score matching;

ec: expert classfication.

#### Example for make data
The input file is placed at ./resource/facebook, and output file will be placed at ./data/facebook

To transform data to SAE graph

./bin/sae -i ./resource/facebook.txt -o ./data/facebook -t md

#### Example for make tencent data
The input file is placed at /tmp/tencent8.graph,and output file will be placed at ./data/tencent8

To transform data to SAE graph

./bin/sae –i /tmp/tencent8.graph -o ./data/tencent8 -t mt

#### Example for influence maximization
The input file is placed at ./data/facebook

To run influence maximization with constant edge weight as 0.5 and number of seed users as 10:

./bin/sae -i ./data/facebook -t im -k 10 -w const -c 0.5

#### Example for dynamicMST
The input file is placed at ./data.txt

case 1:
data.txt contains vertex number n, edge number m and all the edges of the graph:
n m
x1 y1 w1
x2 y2 w2
...

./bin/sae -i ./data.txt -t dm

case 2:
data.txt contains only all the edges of the graph:
x1 y1 w1
x2 y2 w2
...

./bin/sae -i ./data.txt -t dmraw

case 3:
data.txt contains the edges of the graph, but without weight. Program will give a random weight each edge:
x1 y1
x2 y2
...

./bin/sae -i ./data.txt -t dmrawnw


The output file is placed at ./output/dynamicMST.txt
first line is the algorithm executed time
then contains nodes number, edges number and total edge value of the MST
after all the edges' infomation of MST, the last line is the total time of this execution period including output time




#### Example for Community Detection
The input file is placed at ./data/facebook, and output file will be placed at ./output/community_detection or ./output/community_detection_sampling

To run community detection with community number as 5 and aglorithm 4:

./bin/sae -i ./data/facebook -o ./output -t cd -k 5 -r 4

-r 1 means using Girvan-Newman aglorithm , which runs pretty slowly. This algorithm is not recommended when the network has more than 1000 nodes.

-r 2 means using label propagation aglorithm, and k should not be appointed

-r 3 means using louvain method, and k should not be appointed

-r 4 means using k community core, which don't have extra parameters except k

To run community detection sampling algorithm based on Girvan-Newman, with community number as 5 and sample probability as 0.1:

./bin/sae -i ./data/facebook -o ./output -t cs -r 1 -k 5 -p 0.1

-r 1 means speeding up Girvan-Newman aglorithm

-r 2 means speeding up label propagation aglorithm, and k should not be appointed

-r 3 means speeding up louvain method, and k should not be appointed

-r 4 means speeding up k community core aglorithm

#### Example for Network Embedding
The input file is placed at ./data/karate, and output file will be placed at ./output/network_embedding,
and label file is placed at ./resource/karate_label.txt

To run network embedding with aglorithm 1:

./bin/sae -i ./data/karate -o ./output/network_embedding -t ne -r 1

-r 1 means using Deepwalk

-r 2 means using Node2Vec

-r 3 means using LINE

To test network embedding on multi-label classfication task:

./bin/sae -i ./data/karate -o ./output/network_embedding -t te -q ./resource/karate_label.txt

#### Example for SimRank
The input file is placed at ./data/facebook, and output file will be placed at ./output/simrank

To query node 1's Top 50 similar nodes with using Partial Sums Memoization algorithm and weaken factor is 0.8 

./bin/sae -i ./data/facebook -o ./output -t sr -r 0 -c 0.8 -s 1 -k 50

To run query node 2's Top 20 similar nodes with using Random Walk algorithm 

./bin/sae -i ./data/facebook -o ./output -t sr -r 1 -s 2 -k 20

To query Top 20 similar nodes of multiple nodes, which user defined in query file "./output/query.txt"

./bin/sae -i ./data/facebook -o ./output-t sr -r 1 -q  ./output/query.txt -k 20

##### Example for Propensity Score Matching
./bin/sae -i ./data/expert -t psm

3
##### Example for Expert classfication
./bin/sae -i ./data/expert -t ec

./resource/dm-experts.txt
#### Basic Social Analysis Tools
./bin/sae -i ./data/fake -t social

then choose tasks listed on the terminate.
