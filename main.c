#include "NEAT.h"
#include "C_Vector/C_Vector.h"

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

float ScoreNOT(t_Context *context, t_Entity *entity){
    float input = 1.0f;
    float output;

    SimEntity(context, entity, &input, &output);
    return 1.0f / fabs(2.0f - output);
}

float ScoreXOR(t_Context *context, t_Entity *entity){
    static const float inputs[4][2] = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}
    };

    static const float outputs[4] = {
        0.0f, 1.0f, 1.0f, 0.0f
    };

    float out;
    float score = 0.0f;
    for (int i = 0; i < 4; ++i){
        SimEntity(context, entity, inputs[i], &out);
        score += fabsf(outputs[i] - out);
    }

    return -score;
}

t_Entity _copy_entity(t_Entity *parent);

void display_entity(t_Entity *entity){

}

int main(){
    t_Context context = (t_Context){
        .num_input = 2, 
        .num_output = 1,
        .population = 5000,
        .reproduction_selection = 0.6f,
        .c1 = 1.0f, .c2 = 1.0f, .c3 = 3.0f, .ct = 2.0f,
        .prob_new_node = 0.01f, 
        .prob_new_link = 0.1f,
        .score_fn = (t_ScoreFunction) ScoreXOR
    };

    InitContext(&context);
    float best_score = -INFINITY;
    context.best_score = best_score;
    
    int num_generations = 5000;
    for (int i = 0; i < num_generations; ++i) {
        NextGeneration(&context);
        
        for (int j = 0; j < context.population; ++j) {
            context.entities[j].score = context.score_fn(&context, context.entities + j);
            if (context.entities[j].score > context.best_score){
                context.best_score = context.entities[j].score;
                context.best_entity = _copy_entity(context.entities + j);
            }
        }

        if (context.best_score > best_score)
            best_score = context.best_score;

        printf("\r%03d%% done highest score: %.10f", (i*100) / num_generations, best_score);
        fflush(stdout);
    }

    putchar('\n');
    printf("score: %.2f\n", context.score_fn(&context, &context.best_entity));
    printf("genome: %lu\n", get_count_array(context.best_entity.genome));
    printf("neurons: %lu\n", get_count_array(context.best_entity.neurons));

    for (int i = 0; i < (int) get_count_array(context.best_entity.genome); ++i){
        if (context.best_entity.genome[i].enabled)
            printf("%d -> %d : %.2f\n", 
                 context.best_entity.genome[i].node_in, context.best_entity.genome[i].node_out, context.best_entity.genome[i].weight
            );
    }

    static const float inputs[4][2] = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}
    };

    float out;
    for (int i = 0; i < 4; ++i){
        SimEntity(&context, &context.best_entity, inputs[i], &out);
        printf("%.0f ^ %.0f = %.3f\n", inputs[i][0], inputs[i][1], out);
    }

    DestrContext(&context);
}
