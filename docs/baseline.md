# Baseline Agent Performance Analysis

## Overall Performance and Baseline Comparison in Simple Mode
The agents show lower performance in creative attempts and Error recognition compared to other dimensions.

| Agent Name | Task Progress | Action Control | Error Recognition | Creative Attempts | Task Efficiency | Material Usage |
|------------|---------------|----------------|----------------------------------|-------------------|-----------------|------------------------------|
| steve-1    | 28.3%         | 25.3%          | 9.5%                             | 5.5%              | 20.7%           | 33.6%                        |
| vpt-bc     | 25.6%         | 23.4%          | 8.9%                             | 5.8%              | 18.5%           | 31.4%                        |
| vpt-rl     | 22.5%         | 21.7%          | 6.5%                             | 4.9%              | 16.4%           | 28.4%                        |

## Performance Degradation in Hard Mode
In hard mode, agents are more easily distracted, leading to performance degradation.

| Agent Name | Task Progress | Action Control | Error Recognition | Creative Attempts | Task Efficiency | Material Usage |
|------------|---------------|----------------|----------------------------------|-------------------|-----------------|------------------------------|
| steve-1    | 23.1%         | 22.6%          | 6.9%                             | 6.0%              | 16.6%           | 24.5%                        |
| vpt-bc     | 23.0%         | 21.7%          | 6.2%                             | 6.0%              | 15.6%           | 25.0%                        |
| vpt-rl     | 21.0%         | 20.9%          | 5.0%                             | 4.6%              | 14.4%           | 23.7%                        |

## Comparison of Creative Task and Programmatic Task
Creative tasks show lower performance than programmatic tasks. For example, steve-1 scores 16.9% in creative tasks compared to 45.4% in programmatic tasks.

| Task Type      | Task Progress | Action Control | Error Recognition | Creative Attempts | Task Efficiency | Material Usage |
|----------------|---------------|----------------|----------------------------------|-------------------|-----------------|------------------------------|
| Creative Task  | 16.9%         | 18.9%          | 6.4%                             | 2.8%              | 11.9%           | 21.4%                        |
| Programmatic Task | 45.4%        | 35.0%          | 14.2%                            | 9.6%              | 33.8%           | 53.6%                        |
| All Task Averages | 28.3%        | 25.3%          | 9.5%                             | 5.5%              | 20.7%           | 33.6%                        |

## Performance on Common Tasks vs. Expanded Task Set
When the task set expands from 30 common tasks (in MCU paper) to 80 tasks, the average performance of agents decreases.

| Agent Name | 30 Common Tasks | 80 Standard Tasks |
|------------|----------------------------------------|--------------------------------------------------|
| vpt-bc     | 21.9%                                  | 16.7%                                            |
| vpt-rl     | 22.9%                                  | 18.9%                                            |
| steve-1    | 27.7%                                  | 20.4%                                            |