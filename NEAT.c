#define _GNU_SOURCE 1 /* for qsort_r */
#include "NEAT.h"
#include "C_Vector.h"
#include <raylib.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define ABS(a)   ((a) < 0 ? -(a) : (a))

t_Entity _make_entity(const t_Context *context){
    t_Entity new_entity;
    new_entity.genome = alloc_array(sizeof(t_ConnectionGene), context->num_input + context->num_output);
    /* to avoid cost of pbacking */
    get_count_array(new_entity.genome) = context->num_input + context->num_output;

    t_ConnectionGene new_gene = (t_ConnectionGene){
        .node_out = context->num_input,
        .weight = 1.0f,
        .enabled = true,
        .innov_num = 0,
        .is_output_node = false
    };

    /* connections from all input nodes to first output node */
    for (new_gene.node_in = 0; new_gene.node_in < context->num_input; ++new_gene.node_in)
        new_entity.genome[new_gene.node_in] = new_gene;

    new_entity.neurons = alloc_array(sizeof(t_Neuron), context->num_input + context->num_output);

    get_count_array(new_entity.neurons) = context->num_input + context->num_output;

    return new_entity;
}

void InitContext(t_Context *context){
    context->entites = malloc(sizeof(t_Entity) * context->population);

    for (int i = 0; i < context->population; ++i)
        context->entites[i] = _make_entity(context);

    context->innov_num = 0;
}

/* platform-specific qsort_r bs */
#ifdef __APPLE__
    typedef  int(*t_compare_fn)(void*, const void*, const void*);
    #define QSORT_R_WRAPPER(base, nmemb, size, context, compare_fn)\
        qsort_r(base, nmemb, size, context, compare_fn)
int _compare_gene_layer(t_Neuron *neurons, const t_ConnectionGene *gene1, const t_ConnectionGene *gene2)

#else
typedef int(*t_compare_fn)(const void*, const void*, void*)
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

/* calculates outputs based on inputs using an entities genotype */
void SimEntity(t_Context *context, t_Entity *entity, float *inputs, float *outputs){
    memset(entity->neurons, 0, sizeof(t_Neuron) * get_count_array(entity->neurons));

    /* find the nummber of inputs to each neuron */
    for (int i = 0; i < (int) get_count_array(entity->genome); ++i)
        if (entity->genome[i].enabled)
            entity->neurons[entity->genome[i].node_out].num_in++;

    /* assign input values to input neurons */
    for (int i = 0; i < context->num_input; ++i) {
        entity->neurons[i].value  = inputs[i];
        entity->neurons[i].num_in = 1;
        entity->neurons[i].layer = 0;
    }

    /* assign layers to each neuron */
    int max_layer = 0;
    for (int i = 0; i < (int) get_count_array(entity->genome); ++i) {
        if (entity->genome[i].enabled && entity->neurons[entity->genome[i].node_out].layer < entity->neurons[entity->genome[i].node_in].layer) {
            entity->neurons[entity->genome[i].node_out].layer = entity->neurons[entity->genome[i].node_in].layer + 1;
            
            if (max_layer < entity->neurons[entity->genome[i].node_out].layer)
                max_layer = entity->neurons[entity->genome[i].node_out].layer;
        }
    }

    /* sort connections by input layer, input neuron, output neuron */
    QSORT_R_WRAPPER(
        entity->genome, get_count_array(entity->genome), sizeof(t_ConnectionGene), 
        entity->neurons, (t_compare_fn)_compare_gene_layer
    );

    int connection_num = 0;
    for (int layer_num = 1; layer_num <= max_layer; ++layer_num){
        /* update the neurons that have inputs in the current layer */
        while (entity->genome[connection_num].enabled && entity->neurons[entity->genome[connection_num].node_in].layer == layer_num){
            entity->neurons[entity->genome[connection_num].node_out].value += 
                entity->neurons[entity->genome[connection_num].node_in].value * entity->genome[connection_num].weight;
            ++connection_num;
        }
    }

    /* copy the output neuron data */
    for (int i = get_count_array(entity->neurons) - context->num_output; i < (int) get_count_array(entity->neurons); ++i)
        outputs[i] = entity->neurons[i].value;
}

/* generates a random number that trends towards 0 */
float _box_muller_transform(float sigma){
    float u1 = (rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = (rand() + 1.0f) / ((float)RAND_MAX + 2.0f);

    return (sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI * u2)) * sigma;
}

void _add_entity_link(t_Context *context, t_Entity *entity){
    int new_link_in;
    int new_link_out;

    while (true){
        new_link_in  = rand() % get_count_array(entity->neurons);
        new_link_out = rand() % get_count_array(entity->neurons);

        /* make sure connection is forward-facing */
        if (entity->neurons[new_link_in].layer >= entity->neurons[new_link_out].layer)
            continue;

        /* check if the link already exists */
        for (int i = 0; i < (int) get_count_array(entity->genome); ++i)
            if (entity->genome[i].node_in == new_link_in && entity->genome[i].node_out == new_link_out)
                continue;

        break;
    }
    
    /* add the new link */

    t_ConnectionGene new_link = (t_ConnectionGene){
        .node_in = new_link_in,
        .node_out = new_link_out,
        .weight = 1.0f,
        .enabled = true,
        .innov_num = ++context->innov_num,
        .is_output_node = false
    };

    pback_array(&entity->genome, &new_link);
}

void _add_entity_neuron(t_Context *context, t_Entity *entity){
    /* the link that will be split */
    int link = rand() % get_count_array(entity->genome);

    int node_in = entity->genome[link].node_in;
    int node_out = entity->genome[link].node_out;

    /* add the new nueron */
    t_Neuron empty_neuron = (t_Neuron){0};
    pback_array(&entity->neurons, &empty_neuron);

    int new_neuron = get_count_array(entity->neurons) - 1;

    entity->genome[link].enabled = false;

    /* add the two new links */

    t_ConnectionGene new_link = (t_ConnectionGene){
        .node_in = node_in,
        .node_out = new_neuron,
        .weight = 1.0f,
        .enabled = true,
        .innov_num = ++context->innov_num,
        .is_output_node = false
    };
    pback_array(&entity->genome, &new_link);

    new_link.node_in = new_neuron;
    new_link.node_out = node_out;
    pback_array(&entity->genome, &new_link);
}

/* assumes that neurons for the entity are initialized */
void MutateEntity(t_Context *context, t_Entity *entity){
    /* mutate connection weights */
    for (int i = 0; i < (int) get_count_array(entity->genome); ++i)
        if (!(rand() % 20)){
            entity->genome[i].weight += _box_muller_transform(0.1f);

            if (entity->genome[i].weight > 1.0f)
                entity->genome[i].weight = 1.0f; else
            if (entity->genome[i].weight < -1.0f)
                entity->genome[i].weight = -1.0f;
        }

    /* add links */
    if ((rand() / (float)RAND_MAX) <= context->prob_new_link)
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
    qsort(parent1->genome, get_count_array(parent1->genome), sizeof(t_ConnectionGene), (int(*)(const void*, const void*))_compare_innov_num);
    qsort(parent2->genome, get_count_array(parent2->genome), sizeof(t_ConnectionGene), (int(*)(const void*, const void*))_compare_innov_num);

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

    for (int i = 0, j = 0; i < (int) get_count_array(short_genome->genome); ++i, ++j){
        if (short_genome->genome[i].innov_num == long_genome->genome[j].innov_num)
            pback_array(&child->genome, short_genome->genome + i); else

        if (short_genome->genome[i].innov_num < long_genome->genome[j].innov_num) {
            if (parents_are_equal && rand() % 2)
                    pback_array(&child->genome, short_genome->genome + i);  else 
            if (fit_parent == short_genome)
                pback_array(&child, short_genome->genome + i);
                
            ++i;
        } else
        if (long_genome->genome[j].innov_num < short_genome->genome[i].innov_num) {
            if (parents_are_equal && rand() % 2)
                    pback_array(&child, long_genome->genome + j);  else 
            if (fit_parent == long_genome)
                pback_array(&child, long_genome->genome + j);
                
            ++j;
        }
    }

    if (fit_parent == long_genome)
        for (int i = get_count_array(short_genome->genome) - 1; i < (int) get_count_array(long_genome->genome); ++i)
            pback_array(&child->genome, long_genome->genome + i);
}

float GeneSimilarity(t_Context *context, t_Entity *e1, t_Entity *e2){
    /* sort genomes by innovation number */
    qsort(e1->genome, get_count_array(e1->genome), sizeof(t_ConnectionGene), (int(*)(const void*, const void*))_compare_innov_num);
    qsort(e2->genome, get_count_array(e2->genome), sizeof(t_ConnectionGene), (int(*)(const void*, const void*))_compare_innov_num);

    t_Entity *long_genome, *short_genome;
    if (get_count_array(e1->genome) > get_count_array(e2->genome)){
        long_genome  = e1;
        short_genome = e1;
    }
    else {
        long_genome  = e1;
        short_genome = e2;
    }

    int num_disjoint = 0;

    int i = 0, j = 0, num_matching = 0;
    float sum_matching = 0.0f;
    for (; i < (int) get_count_array(short_genome->genome); ++i, ++j){
        if (short_genome->genome[i].innov_num < long_genome->genome[j].innov_num){
            ++i;
            ++num_disjoint;
        } else
        if (long_genome->genome[j].innov_num < short_genome->genome[i].innov_num){
            ++j;
            ++num_disjoint;
        }
        else {
            sum_matching += ABS(short_genome->genome[i].weight - long_genome->genome[j].weight);
            ++num_matching;
        }
    }

    int num_excess = get_count_array(long_genome->genome) - (j + 1);

    return ((float)num_excess * context->c1 + (float)num_disjoint * context->c2) / (float)get_count_array(long_genome->genome) + context->c3 * (sum_matching / (float)num_matching);
}

void SpeciateEntity(t_Context *context, t_Entity *entity){
    while(true) {
        t_Entity *other = context->entites + rand() % context->population;

        if (GeneSimilarity(context, entity, other) < context->ct){
            if (other->species != 0)
                entity->species = other->species;
            else {
                context->species_num++;

                entity->species = 
                other->species = context->species_num;
            }
            return;
        }
    }
}

/* sort ascending by species, descending by score */
int _compare_entity_species(t_Entity *e1, t_Entity *e2){
    int res = e1->species - e2->species;
    if (res)
        return res;
    else
        return e2->score - e1->score;
}

void NextGeneration(t_Context *context){
    float mean_fitness = 0.0f;
    for (int i = 0; i < context->population; ++i){
        mean_fitness += context->entites[i].score = context->score_fn(context->entites + i);
        context->entites[i].species = 0;
    }

    mean_fitness /= (float)context->population;
    
    int *species_pops = calloc(context->species_num + 1, sizeof(int));
    
    for (int i = 0; i < context->population; ++i) {
        SpeciateEntity(context, context->entites + i);
        species_pops[context->entites[i].species]++;
    }

    for (int i = 0; i < context->population; ++i)
        context->entites[i].score /= species_pops[context->entites[i].species];

    /* sort the entities by species */
    qsort(context->entites, context->population, sizeof(t_Entity), (int(*)(const void*, const void*))_compare_entity_species);

    int *new_species_pops = malloc(sizeof(int) * (context->species_num + 1));
    int *species_end      = malloc(sizeof(int) * (context->species_num + 1));

    for (int i = 1, j = 0; i <= context->species_num; ++i){
        float species_fitness_sum = 0.0f;
        
        for (; context->entites[j].species == i; ++j)
            species_fitness_sum += context->entites[j].score;
        
        species_end[i] = j;

        new_species_pops[i] = species_fitness_sum / mean_fitness;
    }

    t_Entity *new_entities = alloc_array(sizeof(t_Entity), context->population);

    t_Entity new_entity;

    /* NOTE: implement crossing over selected entities and new population */
    for (int i = 1, j = 0; i <= context->species_num; ++i){
        int keep_species = roundf((float)(species_end[i] - j) * context->reproduction_selection) + j;

        for (int k = 0; k < new_species_pops[i]; ++k){
            
        }

        j = species_end[i];
    }


}