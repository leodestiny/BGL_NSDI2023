#include "lru.h"

LRU::LRU(CacheNode ** cm): cache_map(cm)
{
	head = new CacheNode;
	tail = new CacheNode;
	head->next = tail;
	head->prev = NULL;

	tail->prev = head;
	tail->next = NULL;
}

LRU::~LRU()
{
	delete head;
	delete tail;
	delete [] cache_map;
}


void LRU::unlink(CacheNode * tmp)
{
		// unlink from previous position
		tmp->next->prev = tmp->prev;
		tmp->prev->next = tmp->next;
}

void LRU::move_to_head(CacheNode * tmp)
{
		// move to head
		head->next->prev = tmp;
		tmp->next = head->next;
		tmp->prev = head;
		head->next = tmp;
}


int64_t LRU::access(int64_t key)
{
	CacheNode * tmp = cache_map[key];
	if(tmp == NULL)
		return -1;

	if(head->next != tmp){
		unlink(tmp);
		move_to_head(tmp);
	}
	return tmp->value;
}

void LRU::insert(int64_t key, int64_t value)
{
	CacheNode * tmp = new CacheNode;
	tmp->key = key;
	tmp->value = value;

	move_to_head(tmp);
	cache_map[key] = tmp;
}

int64_t LRU::evict_and_insert(int key)
{
	CacheNode * tmp = tail->prev;
	cache_map[tmp->key] = NULL;
	tmp->key = key;
	cache_map[key] = tmp;

	unlink(tmp);
	move_to_head(tmp);
	
	return tmp->value;

}




