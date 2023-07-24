#include <iostream>
struct CacheNode
{
	int64_t key;
	int64_t value;
	CacheNode * next, *prev;
};

class LRU
{
private:
	CacheNode ** cache_map;
	CacheNode * head;
	CacheNode * tail;


public:
	LRU(CacheNode ** cm);
	~LRU();
	void unlink(CacheNode* tmp);
	void move_to_head(CacheNode *tmp);
	int64_t access(int64_t key);
	void insert(int64_t key, int64_t value);
	int64_t evict_and_insert(int key);

};
