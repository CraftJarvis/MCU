# Baseline Agent Performance Analysis

## Overall Performance and Baseline Comparison in Simple Mode
The agents show lower performance in creative attempts and Error recognition compared to other dimensions.

| Agent Name | Task Progress | Action Control | Error Recognition | Creative Attempts | Task Efficiency | Material Usage |
|------------|---------------|----------------|-------------------|-------------------|-----------------|----------------|
| steve-1    | 31.4%         | 31.9%          | 13.1%             | 6.4%              | 23.2%           | 35.6%          |
| vpt-bc     | 29.1%         | 29.0%          | 11.8%             | 6.2%              | 21.3%           | 33.8%          |
| vpt-rl     | 25.9%         | 26.2%          | 8.8%              | 5.2%              | 18.9%           | 31.3%          |
| jarvis-vla | 25.6%         | 27.8%          | 9.3%              | 5.5%              | 18.3%           | 30.5%          |

## Performance Degradation in Hard Mode
In hard mode, agents are more easily distracted, leading to performance degradation.

| Agent Name | Task Progress | Action Control | Error Recognition | Creative Attempts | Task Efficiency | Material Usage | 
| --- | --- | --- | --- | --- | --- | --- | 
| steve-1 | 23.1% | 22.6% | 6.9% | 6.0% | 16.6% | 24.5% | 
| vpt-bc | 23.0% | 21.7% | 6.2% | 6.0% | 15.6% | 25.0% | 
| vpt-rl | 21.0% | 20.9% | 5.0% | 4.6% | 14.4% | 23.7% | 
| jarvis-vla | 20.9% | 22.3% | 5.5% | 4.3% | 14.5% | 22.3% | 


## Comparison of Creative Task and Programmatic Task
Creative tasks show lower performance than programmatic tasks. For example, Steve-1 suffers a severe performance degradation of task progress, with a decline of 15.8% in creative tasks.

| Task Type      | Task Progress | Action Control | Error Recognition | Creative Attempts | Task Efficiency | Material Usage |
|----------------|---------------|----------------|----------------------------------|-------------------|-----------------|------------------------------|
| Programmatic Task | 38.4%        | 36.5%          | 16.3%                            | 7.7%              | 27.7%           | 43.8%                        |
| Creative Task  | 22.6%         | 26.2%          | 9.1%                             | 4.7%              | 17.7%           | 25.4%                        |
| Performance Drop | -15.8% | -10.3% | -7.2% | -3.0% | -10.0% | -18.4% |