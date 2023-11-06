#include <cstdint>

extern "C" void c_calculate(
    uint8_t* flattened_distance, const int a_length,
    const bool* flattened_adjacency,
    const int max_distance
    ){

    bool* _flattened_adjacency = new bool[a_length*a_length];
    uint8_t* _flattened_distance = new uint8_t[a_length*a_length];
    bool* _has_paths = new bool[a_length*a_length];
    bool* _has_paths_new = new bool[a_length*a_length];

    for (int i = 0; i < a_length*a_length; i++){
        _flattened_adjacency[i] = flattened_adjacency[i];
        _flattened_distance[i] = _flattened_adjacency[i];
        _has_paths[i] = _flattened_adjacency[i];
    }

    int path_length = 2;
    bool has_distance_changed;
    while (path_length < max_distance + 1){
        for (int i = 0; i < a_length*a_length; i++){
            _has_paths_new[i] = false;
        }

        has_distance_changed = false;
        for (int u = 0; u < a_length; u++){
            for (int v = 0; v < a_length; v++){
                for (int w = 0; w < a_length; w++){
                    // check for at least one non-zero in the row x column product
                    if (_has_paths[u*a_length+w] && _flattened_adjacency[w*a_length+v]){
                        _has_paths_new[u*a_length+v] = true;
                        if (_flattened_distance[u*a_length+v] == 0){
                            _flattened_distance[u*a_length+v] = path_length;
                            has_distance_changed = true;
                        }
                        break;
                    }
                }
            }
        }

        if (!has_distance_changed){
            break;
        }

        for (int i = 0; i < a_length*a_length; i++){
            _has_paths[i] = _has_paths_new[i];
        }

        path_length++;
    }

    for (int i = 0; i < a_length*a_length; i++){
        flattened_distance[i] = _flattened_distance[i];
    }
}
