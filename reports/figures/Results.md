### Model Performance Comparison

#### Table 1
| Model                     | Pipeline | Split Type | Precision | Recall | F1-Score |
|---------------------------|----------|------------|-----------|--------|----------|
| **Sofficite**             | —        | —          | 0.9338    | 0.8203 | 0.8734   |
| **Gemma2:9b-instruct-q8** | SAT      | Sentence   | 0.6876    | 0.8008 | 0.7399   |
| **Gemma2:9b-instruct-fp16** | SAT    | Sentence   | 0.6801    | 0.7955 | 0.7333   |
| **Sonnet**                | SAT      | Complete   | 0.6370    | 0.7667 | 0.6958   |
| **Gemma2:9b-instruct-fp16** | RAG    | Sentence   | 0.6665    | 0.6743 | 0.6704   |
| **Gemma2:9b-instruct-q8** | RAG      | Sentence   | 0.6646    | 0.6743 | 0.6694   |
| **Llama3.1:70b-instruct-q4** | SAT    | Paragraph  | 0.5477    | 0.7970 | 0.6493   |
| **Gemma2:9b-instruct-q8** | SAT      | Paragraph  | 0.5694    | 0.7376 | 0.6426   |
| **Llama3:70b**            | SAT      | Sentence   | 0.6818    | 0.6057 | 0.6415   |
| **Gemma2:9b-instruct-fp16** | SAT    | Paragraph  | 0.5602    | 0.7473 | 0.6404   |

Comparison of the performances of different models on the SoftCite dataset v2.

#### Table 2
| Model                     | Pipeline | Split Type | Precision | Recall | F1-Score |
|---------------------------|----------|------------|-----------|--------|----------|
| **Llama3:70b**            | SAT      | Sentence   | 0.7326    | 0.7452 | 0.7388   |
| **Gemma2:9b-instruct-fp16** | RAG    | Sentence   | 0.7464    | 0.6678 | 0.7049   |
| **Gemma2:9b-instruct-fp16** | SAT    | Sentence   | 0.6671    | 0.6943 | 0.6804   |
| **Llama3:70b**            | RAG      | Sentence   | 0.6780    | 0.6644 | 0.6712   |
| **Sonnet**                | RAG      | Sentence   | 0.6689    | 0.6703 | 0.6696   |
| **Gemma2:27b-instruct-fp16** | RAG    | Sentence   | 0.6989    | 0.6376 | 0.6668   |
| **Sonnet**                | SAT      | Sentence   | 0.5711    | 0.7064 | 0.6316   |
| **Gemma2:9b-instruct-fp16** | SAT    | Paragraph  | 0.5775    | 0.6862 | 0.6272   |
| **Sonnet**                | SAT      | Paragraph  | 0.5673    | 0.6404 | 0.6016   |
| **Softcite**              | —        | —          | 0.4330    | 0.4147 | 0.4236   |

Comparison of the performances of different models on the SoMeSci dataset.


#### Table 3
| Model                     | Pipeline | Split Type | Precision | Recall | F1-Score |
|---------------------------|----------|------------|-----------|--------|----------|
| **Llama3:70b**            | SAT      | Sentence   | 0.6971    | 0.7315 | 0.7139   |
| **Sonnet**                | SAT      | Sentence   | 0.6182    | 0.7335 | 0.6709   |
| **Sonnet**                | SAT      | Paragraph  | 0.6098    | 0.6926 | 0.6486   |
| **Gemma2:9b-instruct-fp16** | SAT    | Sentence   | 0.6195    | 0.6532 | 0.6359   |
| **Llama3:70b**            | RAG      | Sentence   | 0.7187    | 0.5659 | 0.6332   |
| **Gemma2:9b-instruct-fp16** | SAT    | Paragraph  | 0.5986    | 0.6702 | 0.6324   |
| **Sonnet**                | RAG      | Sentence   | 0.6313    | 0.6026 | 0.6166   |
| **Gemma2:9b-instruct-fp16** | RAG    | Sentence   | 0.6638    | 0.5197 | 0.5830   |
| **Llama3:8b**             | SAT      | Paragraph  | 0.5176    | 0.6612 | 0.5807   |
| **Softcite**              | —        | —          | 0.3350    | 0.3220 | 0.3284   |

Comparison of the performances of different models on the SoFAIR dataset.
