// clone from https://github.com/Aj163/LFU.git
#include "lfu.h"
#include <cstdlib>
#include <cstring>


LFU::LFU(dataNode ** mm)
{
	freq_head = new freqNode;
	freq_head->next = NULL;
	freq_head->prev = NULL;

	m = mm;
	
}

LFU::~LFU()
{
	// delete entire double linked list?
	freq_head->next = NULL;
	delete freq_head;
	delete [] m;
}

freqNode *LFU::newFreqNode(int frequency, freqNode *next, freqNode *prev)
{
	freqNode *obj = new freqNode;
	obj->freq = frequency;
	obj->next = next;
	obj->head = NULL;
	obj->prev = prev;
	if(next != NULL)
		next->prev = obj;
	prev->next = obj;
	return obj;
}

void LFU::delFreqNode(freqNode *tmp)
{
	tmp->prev->next = tmp->next;
	if(tmp->next !=NULL)
		tmp->next->prev = tmp->prev;
	delete tmp;
}

dataNode *LFU::newDataNode(int key, int value, freqNode *parent)
{
	dataNode *obj = new dataNode;
	obj->key = key;
	obj->value = value;
	obj->next = parent->head;
	if(parent->head !=NULL)
		parent->head->prev = obj;
	obj->parent = parent;
	parent->head = obj;
	obj->prev = NULL;
	return obj;
}

void LFU::unlinkDataNode(dataNode *tmp)
{
	if(tmp->prev !=NULL)
		tmp->prev->next = tmp->next;
	else
		tmp->parent->head = tmp->next;

	if(tmp->next !=NULL)
		tmp->next->prev = tmp->prev;

	if(tmp->parent->head == NULL)
		delFreqNode(tmp->parent);
}

int64_t LFU::access(int64_t key)
{
	if(m[key] == NULL)
	{
		return -1;
	}

	dataNode *tmp = m[key];
	freqNode * freq_node = tmp->parent;
	int cur_freq = freq_node->freq;
	freqNode *next_freq_node = freq_node->next;

	if(next_freq_node == NULL || next_freq_node->freq != cur_freq +1)
		next_freq_node = newFreqNode(cur_freq +1, next_freq_node, tmp->parent);

	// unlink tmp datanode in original freq list
	unlinkDataNode(tmp);

	tmp->next = next_freq_node->head;
	if(next_freq_node->head != NULL)
		next_freq_node->head->prev = tmp;
	next_freq_node->head = tmp;
	tmp->parent  = next_freq_node;
	tmp->prev = NULL;

	return tmp->value;

}

void LFU::insert(int64_t key, int64_t value)
{
	freqNode *start_freq_node = freq_head->next;

	if(start_freq_node == NULL || start_freq_node->freq !=1)
		start_freq_node = newFreqNode(1, start_freq_node, freq_head);


	dataNode *tmp = newDataNode(key, value, start_freq_node);
	m[key] = tmp;
}

int64_t LFU::evict()
{
	if(freq_head->next == NULL)
	{
		cout<<"The set is empty\n";
		return inf;
	}
	
	dataNode *tmp = freq_head->next->head;
	int value = tmp->value;
	int key = tmp->key;
	unlinkDataNode(tmp);
	m[key] = NULL;
	delete tmp;
	return value;
}

void LFU::printLfu()
{
	freqNode *f = freq_head->next;
	dataNode *tmp;

	cout<<"\n";
	while(f!=NULL)
	{
		cout<<"Frequency "<<f->freq<<" : ";
		tmp = f->head;
		while(tmp!=NULL)
		{
			cout<<"key: " << tmp->key << "value: " << tmp->value <<" ";
			tmp = tmp->next;
		}
		cout<<endl;
		f = f->next;
	}
	cout<<endl;
}
