# A Kernel Scheduler Based on KnapSack

## Profiling

### Profiling Results

## Offline Scheduling

### Scheduling Strategy

A scheduling strategy is generated from the profiling results. 

```json
{
    "scheduling_slice(ms)": 10, 
    "kernel_candidates": [ <kernels for scheduling>
        {
            "label": "kernel0", 
            "nblock": <number of blocks>, 
            "nkernelslice": <number of kernel slices>, 
            ...
        }, 
        {
            "label": "kernel1", 
            ...
        }, 
        ...
    ], 
    "scheduling_plan": [ <assigned kernel slice in each scheduling slice>
        [ <in scheduling slice 0>
            0, 0, 1, 1, ... <each kernel ID indicates a kernel slice of that kernel is launched in this scheduling slice>
        ], 
        [
            ...
        ], 
        ...
    ]
}
```
