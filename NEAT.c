#define _GNU_SOURCE /* for qsort_r */
#include "NEAT.h"
#include "C_Vector/C_Vector.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define ABS(a)   ((a) < 0 ? -(a) : (a))

t_Entity _make_entity(const t_Context *context){
    t_Entity new_entity;
    new_entity.genome = alloc_array(sizeof(t_ConnectionGene), context->num_input);

    t_ConnectionGene new_gene = (t_ConnectionGene){
        .node_in   = 0,
        .node_out  = context->num_input,
        .weight    = 1.0f,
        .enabled   = true,
        .innov_num = 0,
    };

    /* connections from the all input nodes to first output node */
    for (new_gene.node_in = 0; new_gene.node_in < context->num_input; ++new_gene.node_in)
        pback_array(&new_entity.genome, &new_gene);

    /* bias node -> output */
    new_gene.node_in = context->num_input + context->num_output;
    pback_array(&new_entity.genome, &new_gene);

    new_entity.neurons = alloc_array(sizeof(t_Neuron), context->num_input + context->num_output + 1);

    /* i coulddd just assign the count here but that scares me - i feel like something is going wrong */
    t_Neuron new_neuron = {0};
    for (int i = 0; i < context->num_input + context->num_output + 1; ++i)
        pback_array(&new_entity.neurons, &new_neuron);

    new_entity.species = 0;

    return new_entity;
}

t_Entity _copy_entity(t_Entity *parent) {
    t_Entity new_entity;

    new_entity.genome = alloc_array(sizeof(t_ConnectionGene), get_count_array(parent->genome));
    get_count_array(new_entity.genome) = get_count_array(parent->genome);
    memcpy(new_entity.genome, parent->genome, get_count_array(new_entity.genome) * sizeof(t_ConnectionGene));

    new_entity.neurons = alloc_array(sizeof(t_Neuron), get_count_array(parent->neurons));
    get_count_array(new_entity.neurons) = get_count_array(parent->neurons);
    memcpy(new_entity.neurons, parent->neurons, get_count_array(new_entity.neurons) * sizeof(t_Neuron));

    new_entity.species = parent->species;

    new_entity.score = parent->score;

    return new_entity;
}

void _destr_entity(t_Entity *entity){
    free_array(entity->genome, NULL);
    free_array(entity->neurons, NULL);
}

void InitContext(t_Context *context){
    context->entities = malloc(sizeof(t_Entity) * context->population);

    for (int i = 0; i < context->population; ++i) {
        context->entities[i] = _make_entity(context);
        context->entities[i].species = 0;
    }

    context->innov_num = 0;
    context->species_num = 0;
    context->best_score = 0.0f;

    srand(time(NULL));
}

void DestrContext(t_Context *context){
    for (int i = 0; i < context->population; ++i)
        _destr_entity(context->entities + i);

    free(context->entities);
}

/* platform-specific qsort_r bs */
#ifdef __APPLE__
    typedef  int(*t_compare_fn)(void*, const void*, const void*);
    #define QSORT_R_WRAPPER(base, nmemb, size, context, compare_fn)\
        qsort_r(base, nmemb, size, context, compare_fn)
int _compare_gene_layer(t_Neuron *neurons, const t_ConnectionGene *gene1, const t_ConnectionGene *gene2)

#else
typedef int(*t_compare_fn)(const void*, const void*, void*);
    #define QSORT_R_WRAPPER(base, nmemb, size, context, compare_fn)\
        qsort_r(base, nmemb, size, compare_fn, context)

int _compare_gene_layer(const t_ConnectionGene *gene1, const t_ConnectionGene *gene2, t_Neuron *neurons)
#endif
{
    int layer_diff = neurons[gene1->node_in].layer - neurons[gene2->node_in].layer;
    if (layer_diff)
        return layer_diff;
        
    int node_diff = gene1->node_in - gene2->node_in;
    if (node_diff)
        return node_diff;
    
    return gene1->node_out - gene2->node_out;
}

void _assign_neuron_layers(t_Entity *entity, int *out_max_layer){
    *out_max_layer = 0;
    bool updated;
    do {
        updated = false;
        for (int i = 0; i < (int) get_count_array(entity->genome); ++i) {
            int in  = entity->genome[i].node_in,
                out = entity->genome[i].node_out;

            if (entity->neurons[out].layer <= entity->neurons[in].layer) {
                entity->neurons[out].layer = entity->neurons[in].layer + 1;
                
                if (*out_max_layer < entity->neurons[out].layer)
                    *out_max_layer = entity->neurons[out].layer;

                updated = true;
            }
        }
    } while(updated);
}

/**
 * randomly generates a new weight
 * 	based off of the number <old>
 */ 
float _gen_weight(float old){
	float diff = rand() % 1000000;
	diff = tanhf(diff / 1000000);
	diff = rand() & 1 ? diff : -diff;
	return old + diff;
}

/* calculates outputs based on inputs using an entities genotype */
void SimEntity(t_Context *context, t_Entity *entity, const float *inputs, float *outputs){
    memset(
        entity->neurons, 0, 
        sizeof(t_Neuron) * get_count_array(entity->neurons)
    );

    /* find the nummber of inputs to each neuron */
    for (int i = 0; i < (int) get_count_array(entity->genome); ++i)
        if (entity->genome[i].enabled)
            entity->neurons[entity->genome[i].node_out].num_in++;


    int max_layer;
    _assign_neuron_layers(entity, &max_layer);

    /* sort connections by input layer, input neuron, output neuron */
    QSORT_R_WRAPPER(
        entity->genome, get_count_array(entity->genome), sizeof(t_ConnectionGene), 
        entity->neurons, (t_compare_fn)_compare_gene_layer
    );
    
    /* assign input values to input neurons */
    for (int i = 0; i < context->num_input; ++i) {
        entity->neurons[i].value  = inputs[i];
        entity->neurons[i].num_in = 1;
        entity->neurons[i].layer = 0;
    }

    /* assign bias neuron input */
    entity->neurons[context->num_input + context->num_output] = (t_Neuron){
        .layer = 0,
        .value = 1.0f,
        .num_in = 1
    };

    int connection_num = 0;
    for (int layer_num = 0; layer_num < max_layer; ++layer_num){
        /* update the neurons that have inputs in the current layer */
        while (
            connection_num < (int) get_count_array(entity->genome) && 
            entity->neurons[entity->genome[connection_num].node_in].layer == layer_num
        ){
            if (entity->genome[connection_num].enabled)
                entity->neurons[entity->genome[connection_num].node_out].value +=
                    entity->neurons[entity->genome[connection_num].node_in].value * entity->genome[connection_num].weight;
            ++connection_num;
        }

        for (int i = 0; i < (int) get_count_array(entity->neurons); ++i)
            if (entity->neurons[i].layer == layer_num + 1 && i != context->num_input + context->num_output)
                entity->neurons[i].value = tanhf(entity->neurons[i].value);
    }

    /* copy the output neuron data */
    for (int i = context->num_input; i < (context->num_input + context->num_output); ++i)
        outputs[i - context->num_input] = entity->neurons[i].value;
}

void _add_entity_link(t_Context *context, t_Entity *entity){
    int new_link_in;
    int new_link_out;

    int i = 0;
    while (true){
        begin_loop:

        if (i > 100)
            return;
        ++i;

        new_link_in  = rand() % get_count_array(entity->neurons);
        new_link_out = rand() % get_count_array(entity->neurons);

        /* make sure connection is forward-facing */
        if (entity->neurons[new_link_in].layer >= entity->neurons[new_link_out].layer)
            continue;

        /* check if the link already exists */
        for (int i = 0; i < (int) get_count_array(entity->genome); ++i)
            if (entity->genome[i].node_in == new_link_in && entity->genome[i].node_out == new_link_out)
                goto begin_loop;

        break;
    }
    
    /* add the new link */
    t_ConnectionGene new_link = (t_ConnectionGene){
        .node_in = new_link_in,
        .node_out = new_link_out,
        .weight = _gen_weight(0.0f),
        .enabled = true,
        .innov_num = ++context->innov_num,
    };

    pback_array(&entity->genome, &new_link);
}

void _add_entity_neuron(t_Context *context, t_Entity *entity){
    /* the link that will be split */
    int link = 0;
    do
        link = rand() % get_count_array(entity->genome);
    while (!entity->genome[link].enabled);

    int node_in  = entity->genome[link].node_in,
        node_out = entity->genome[link].node_out;

    /* add the new nueron */
    t_Neuron empty_neuron = (t_Neuron){0};
    pback_array(&entity->neurons, &empty_neuron);

    int new_neuron = get_count_array(entity->neurons) - 1;

    entity->genome[link].enabled = false;

    /* add the two new links */

    t_ConnectionGene new_link = (t_ConnectionGene){
        .node_in = node_in,
        .node_out = new_neuron,
        .weight = entity->genome[link].weight,
        .enabled = true,
        .innov_num = ++context->innov_num,
    };
    pback_array(&entity->genome, &new_link);

    new_link.node_in = new_neuron;
    new_link.node_out = node_out;
    new_link.innov_num = ++context->innov_num;
    pback_array(&entity->genome, &new_link);
}

/* assumes that neurons for the entity are initialized */
void MutateEntity(t_Context *context, t_Entity *entity){
    /* mutate connection weights */
    for (int i = 0; i < (int) get_count_array(entity->genome); ++i)
        if (rand() % 16){
           	entity->genome[i].weight = _gen_weight(entity->genome[i].weight); 
            if (entity->genome[i].weight > 1.0f)
                entity->genome[i].weight = 1.0f; else
            if (entity->genome[i].weight < -1.0f)
                entity->genome[i].weight = -1.0f;
        }

    /* add links - cant add a link if there are only 2 neurons */
    if (get_count_array(entity->neurons) > 2 && (rand() / (float)RAND_MAX) <= context->prob_new_link)
        _add_entity_link(context, entity);

    /* add neurons */
    if ((rand() / (float)RAND_MAX) <= context->prob_new_node)
        _add_entity_neuron(context, entity);
}

int _compare_innov_num(t_ConnectionGene *gene1, t_ConnectionGene *gene2){
    return gene1->innov_num - gene2->innov_num;
}

void CrossoverEntities(t_Context *context, t_Entity *parent1, t_Entity *parent2, t_Entity *child){
    *child = _make_entity(context);

    /* sort genomes by innovation number */
    qsort(
        parent1->genome,
        get_count_array(parent1->genome), 
        sizeof(t_ConnectionGene), 
        (int(*)(const void*, const void*))_compare_innov_num
    );

    qsort(
        parent2->genome, 
        get_count_array(parent2->genome), 
        sizeof(t_ConnectionGene), 
        (int(*)(const void*, const void*))_compare_innov_num
    );

    t_Entity *long_genome, *short_genome;
    if (get_count_array(parent1->genome) > get_count_array(parent2->genome)){
        long_genome  = parent1;
        short_genome = parent2;
    }
    else {
        long_genome  = parent2;
        short_genome = parent1;
    }

    t_Entity *fit_parent = (parent1->score > parent2->score) ? parent1 : parent2;
    bool parents_are_equal = parent1->score == parent2->score;

    for (int i = 0, j = 0; i < (int) get_count_array(short_genome->genome) && j < (int) get_count_array(long_genome->genome); ++i, ++j){
        if (short_genome->genome[i].innov_num == long_genome->genome[j].innov_num)
            pback_array(&child->genome, short_genome->genome + i); else
        
        if (short_genome->genome[i].innov_num < long_genome->genome[j].innov_num) {
            if (parents_are_equal && rand() % 2)
                    pback_array(&child->genome, short_genome->genome + i);  else 
            if (fit_parent == short_genome)
                pback_array(&child->genome, short_genome->genome + i);
            ++i;
        } else
        if (long_genome->genome[j].innov_num < short_genome->genome[i].innov_num) {
            if (parents_are_equal && rand() % 2)
                    pback_array(&child->genome, long_genome->genome + j);  else 
            if (fit_parent == long_genome)
                pback_array(&child->genome, long_genome->genome + j);
            ++j;
        }
    }
    if (fit_parent == long_genome && !parents_are_equal) {
        for (int i = get_count_array(short_genome->genome) - 1; i < (int) get_count_array(long_genome->genome); ++i)
            pback_array(&child->genome, long_genome->genome + i);  
    }

    int num_neurons = 0;
    for (int i = 0; i < (int) get_count_array(child->genome); ++i){
        if (child->genome[i].node_out > num_neurons)
            num_neurons = child->genome[i].node_out;
        if (child->genome[i].node_in > num_neurons)
            num_neurons = child->genome[i].node_in;
    }

    free_array(child->neurons, NULL);
    child->neurons = alloc_array(sizeof(t_Neuron), 1);

    t_Neuron empty_neuron = (t_Neuron){0};
    for (int i = 0; i < num_neurons + 1; ++i)
        pback_array(&child->neurons, &empty_neuron);
    
    int garbage;
    _assign_neuron_layers(child, &garbage);
}

float GeneSimilarity(t_Context *context, t_Entity *e1, t_Entity *e2){
    /* sort genomes by innovation number */
    qsort(e1->genome, get_count_array(e1->genome), sizeof(t_ConnectionGene), (int(*)(const void*, const void*))_compare_innov_num);
    qsort(e2->genome, get_count_array(e2->genome), sizeof(t_ConnectionGene), (int(*)(const void*, const void*))_compare_innov_num);

    t_Entity *long_genome, *short_genome;
    if (get_count_array(e1->genome) > get_count_array(e2->genome)){
        long_genome  = e1;
        short_genome = e2;
    }
    else {
        long_genome  = e2;
        short_genome = e1;
    }

    int num_disjoint = 0;

    int i = 0, j = 0, num_matching = 0;
    float sum_diff = 0.0f;
    while (i < (int) get_count_array(short_genome->genome) && j < (int) get_count_array(long_genome->genome)){
        if (short_genome->genome[i].innov_num < long_genome->genome[j].innov_num){
            ++i;
            ++num_disjoint;
        } else
        if (long_genome->genome[j].innov_num < short_genome->genome[i].innov_num){
            ++j;
            ++num_disjoint;
        }
        else {
            sum_diff += ABS(short_genome->genome[i].weight - long_genome->genome[j].weight);
            ++i, ++j;
            ++num_matching;
        }
    }

    int num_excess = (get_count_array(long_genome->genome) - i) + (get_count_array(short_genome->genome) - j);
    float len = get_count_array(long_genome->genome) < 20 ? 1.0f : get_count_array(long_genome->genome);

    return ((float)num_excess * context->c1 + (float)num_disjoint * context->c2) / len + context->c3 * (sum_diff / num_matching);
}

void SpeciateEntity(t_Context *context, t_Entity *entity){
    for (int i = 0; i < context->population; ++i){
        t_Entity *other = context->entities + i;
        if (other != entity && GeneSimilarity(context, entity, other) < context->ct){
            if (other->species != 0) {
                entity->species = other->species;
                return;
            }
        }
    }

    entity->species = ++context->species_num;
}

/* sort ascending by species, descending by score */
int _compare_entity_species(t_Entity *e1, t_Entity *e2){
    int res = e1->species - e2->species;
    if (res)
        return res;
    else
        return e2->score - e1->score;
}

int _compare_entity_score(t_Entity *e1, t_Entity *e2){
    if (e1->score < e2->score) return 1;
    if (e1->score > e2->score) return -1;

    return 0;
}

void NextGeneration(t_Context *context){
    context->species_num = 0;
    for (int i = 0; i < context->population; ++i) {
        context->entities[i].score = context->score_fn(context, context->entities + i);
        context->entities[i].species = 0;
    }
    
    for (int i = 0; i < context->population; ++i)
        SpeciateEntity(context, context->entities + i);

    int *species_pops = calloc(context->species_num + 1, sizeof(int));

    for (int i = 0; i < context->population; ++i) 
        species_pops[context->entities[i].species]++;

    float mean_fitness = 0.0f;
    for (int i = 0; i < context->population; ++i)
        mean_fitness += (context->entities[i].score /= (float)species_pops[context->entities[i].species]);

    mean_fitness /= (float)context->population;

    /* sort the entities by species */
    qsort(
         context->entities, 
         context->population, 
         sizeof(t_Entity), 
         (int(*)(const void*, const void*))_compare_entity_species
    );

    int *new_species_pops = malloc(sizeof(int) * (context->species_num + 1)),
        *species_end      = malloc(sizeof(int) * (context->species_num + 1));

    species_end[0] = 0;

    for (int i = 1, j = 0; i <= context->species_num; ++i){
        float species_fitness_sum = 0.0f;
        
        for (; j < context->population && context->entities[j].species == i; ++j)
            species_fitness_sum += context->entities[j].score;
        
        species_end[i] = j;

        new_species_pops[i] = species_fitness_sum / mean_fitness;
    }

    t_Entity *new_entities = alloc_array(sizeof(t_Entity), context->population);

    t_Entity new_entity;

    for (int i = 1; i <= context->species_num; ++i){
        int keep_species = 
            roundf((float)(species_end[i] - species_end[i - 1]) * context->reproduction_selection + (float)species_end[i - 1]);

        for (int k = 0; k < new_species_pops[i]; ++k){
            int parent  = rand() % ((keep_species - 1) - species_end[i - 1] + 1) + species_end[i - 1];
            // int parent2 = rand() % ((keep_species - 1) - species_end[i - 1] + 1) + species_end[i - 1];

            // CrossoverEntities(context, context->entities + parent, context->entities + parent2, &new_entity);

            new_entity = _copy_entity(context->entities + parent);
            int idk;
            _assign_neuron_layers(&new_entity, &idk);
            MutateEntity(context, &new_entity);

            pback_array(&new_entities, &new_entity);
        }
    }

    if ((int) get_count_array(new_entities) < context->population) {
        qsort(context->entities, context->population, sizeof(t_Entity), (int(*)(const void*, const void*))_compare_entity_score);
        int need = context->population - (int) get_count_array(new_entities);
        for (int i = 0; i < need; ++i) {
            new_entity = _copy_entity(context->entities + i);
            int idk;
            _assign_neuron_layers(&new_entity, &idk);
            pback_array(&new_entities, &new_entity);
        }
    }

    for (int i = 0; i < context->population; ++i)
        _destr_entity(context->entities + i);

    memcpy(context->entities, new_entities, sizeof(t_Entity) * context->population);
    free_array(new_entities, NULL);

    free(species_pops);
    free(species_end);
    free(new_species_pops);
}
