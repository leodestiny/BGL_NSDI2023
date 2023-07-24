// clone from https://github.com/Aj163/LFU.git
#include <unordered_map>
#include <iostream>
#define inf (-1e9)

using namespace std;

#define MAX_NODE 10000000

struct dataNode;
struct freqNode
{
	int64_t freq;
	freqNode *next, *prev;
	dataNode *head; 	//Strating of freq_list
};

struct dataNode
{
	int64_t key;
	int64_t value;
	dataNode *next, *prev;
	freqNode *parent;
};

class LFU
{
private:

	freqNode *freq_head;
	dataNode **m;
	//unordered_map<int, dataNode*> m;

	freqNode *newFreqNode(int frequency, freqNode *next, freqNode *prev);
	void delFreqNode(freqNode *tmp);

	dataNode *newDataNode(int key, int value, freqNode *parent);
	void unlinkDataNode(dataNode *tmp);

public:

	LFU(dataNode ** mm);
	~LFU();

	int64_t access(int64_t key);
	void insert(int64_t key, int64_t value);
	int64_t evict();
	void printLfu();

};


