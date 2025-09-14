#ifndef __NEAT_H__
#define __NEAT_H__

#include <stdbool.h>

typedef struct {
    int   node_in;
    int   node_out;
    float weight;
	int   innov_num;
	bool  enabled;
} t_ConnectionGene;

typedef struct {
    float value;
    int   num_in;
    int   layer;
} t_Neuron;

typedef struct {
    t_ConnectionGene *genome;
    t_Neuron         *neurons;
    float            score;
    void             *entity_data;
	int              species;
} t_Entity;

typedef float (*t_ScoreFunction)(void *context, t_Entity *entity);

typedef struct {
    int                num_input, 
					   num_output,
                       population,
					   species_num,
					   innov_num;
    float              prob_new_node, 
					   prob_new_link,
					   reproduction_selection,
					   c1, c2, c3, ct;
    t_ScoreFunction    score_fn;
    t_Entity           *entities;
    t_Entity           best_entity;
    float              best_score;
} t_Context;
/**
 * Initializes a neural net context
 * @param num_input the number of input neurons in the neural network
 * @param num_output the number of output neurons in the neural network
 * @param population the amount of entities simulated
 * @param reproduction_selection [0:1] percentage of top entities that have chance to reproduce
 * @param prob_new_node [0:1] probability that a new node will be mutated
 * @param prob_new_link [0:1] probability that a new link will be mutated
 * @param score_fn scores the entity based on genotype, higher scores are better
 * @param entities  n/a
 * @param innov_num n/a
 */
void InitContext(t_Context *context);

/**
 * Simulates a neural net based off the entities genome
 * @param context the context of the library being used
 * @param entity  the entity being simulated
 * @param inputs  buffer that is at least context.num_input long that holds the inputs to the neural net
 * @param outputs buffer that is at least context.num_output long that the outputs of the neural net will be copied to
 */
void SimEntity(t_Context *context, t_Entity *entity, const float *inputs, float *outputs);

void NextGeneration(t_Context *context);

void DrawSimulation(t_Context *context);

void DestrContext(t_Context *context);

#endif
