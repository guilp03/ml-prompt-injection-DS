# Prompt Injection Detection is Regime-Dependent: A Deployment-Aware Evaluation with Interpretable Structural Signals

**Autores:** Akindoyin Akinrele e Shreyank N. Gowda (School of Computer Science, University of Nottingham)
**Publicação:** arXiv:2605.26999v1 (26 Mai 2026); também em PMLR v348 — Proceedings of the Fourth UK AI Conference 2026, pp. 12–30.
**Contato:** psyaa28@nottingham.ac.uk / shreyank.narayanagowda@nottingham.ac.uk

---

## 1. Motivação e problema

Prompt injection é uma ameaça crítica à implantação segura de LLMs, mas as abordagens de detecção existentes costumam ser avaliadas em configurações limitadas que não refletem restrições operacionais reais. A literatura acadêmica prioriza métricas agregadas de benchmark (macro-F1, ROC-AUC), que fornecem uma visão global útil mas **não refletem o comportamento em produção**, especialmente sob restrições estritas de falso-positivo.

O artigo reposiciona a detecção de prompt injection como um **problema de classificação orientado à implantação** (deployment-oriented), com duas perspectivas complementares:

- **Ranking:** o detector atribui um escore de risco; o desempenho depende de prompts adversariais pontuarem acima dos benignos ao longo dos thresholds.
- **Deploy:** o detector opera como salvaguarda dentro de um pipeline de LLM e precisa tomar decisões allow/block sujeitas a restrições de falso-positivo.

A distinção é importante porque um modelo com boas métricas globais pode ser inadequado para implantação se bloquear legítimos demais. A questão central não é apenas se prompts adversariais podem ser distinguidos dos benignos em média, mas se a detecção **permanece eficaz sob restrições operacionais** (baixo FPR, generalização OOD, regimes de ataque variados).

---

## 2. Metodologia

### 2.1 Formulação do problema

Detecção binária (benigno = 0, adversarial = 1) avaliada sob duas óticas: qualidade de **ranking** (threshold-free) e qualidade de **decisão de implantação** (thresholded a baixo FPR).

### 2.2 Dados e regimes OOD

**Pool in-distribution (ID)** — composto por três fontes públicas (comparabilidade com trabalhos anteriores, sem geração sintética):
- AdvBench (prompts jailbreak nocivos)
- JailbreakBench (nocivos + benignos tematicamente similares)
- Deepset prompt-injection (ataques de hierarquia de instrução para sistemas RAG/tool)

Após pré-processamento, o pool ID foi particionado em treino/validação/teste:

| Partição | Datasets constituintes | Total | Adversarial | Benigno |
|---|---|---|---|---|
| ID Train (A–C) | AdvBench + JailbreakBench + Deepset PI | 694 | 504 | 190 |
| ID Validation (A–C) | AdvBench + JailbreakBench + Deepset PI | 149 | 108 | 41 |
| ID Test (A–C) | AdvBench + JailbreakBench + Deepset PI | 149 | 109 | 40 |
| Primary OOD | HarmBench + Alpaca Instructions | 768 | 384 | 384 |
| Hard-negative Injection OOD | r1char9 Prompt-to-Prompt Injection + NotInject | 678 | 339 | 339 |
| Standard Injection OOD | Qualifire Prompt-Injection Benchmark | 3.986 | 1.993 | 1.993 |
| **Total** | Todos os datasets incluídos | **6.424** | **3.437** | **2.987** |

**Três regimes OOD fixos (held-out):**
1. **Primary OOD (HarmBench + Alpaca)** — teste de generalização amplo orientado a jailbreak.
2. **Hard-negative Injection OOD (r1char9 + NotInject)** — intencionalmente desafiador: os benignos contêm formulações que imitam ataques, expondo detectores *over-defensive* sob baixo FPR.
3. **Standard Injection OOD (Qualifire)** — benchmark de injeção maior e balanceado, sem lookalikes adversariais.

**Dados OOD foram excluídos** de design de features, ajuste de modelo, seleção de modelo e tuning de threshold — garantindo generalização genuína.

### 2.3 Esplits e repetibilidade

Todos os ajustes (model fitting, seleção de features, tuning de threshold) restritos aos dados ID. Para reduzir dependência de um split favorável, foram usados **3 splits** (A, B, C) com as mesmas proporções/tamanhos mas sementes aleatórias distintas. OOD idêntico em todos os runs.

### 2.4 Pré-processamento

- Normalização Unicode NFKC, strip de whitespace, colapso de espaços repetidos.
- **Sem** pré-processamento agressivo (não remove stopwords nem pontuação) — para não eliminar sinais lexicais/estruturais centrais à detecção.
- Formas canônicas usadas para deduplicação, checagem de conflitos e filtragem de leakage entre ID e OOD.

### 2.5 Famílias de features (escada progressiva)

**1. Lexical**
- TF-IDF (unigramas + bigramas, vocabulário limitado a 20.000 termos) + Regressão Logística. Baseline transparente e reproduzível (lower bound de sinal adversarial recuperável apenas por wording).
- Pequeno conjunto de **flags lexicais binárias** (linguagem de override, referências a system prompt, frases de descumprimento de regras, reassinalamento de papel).

**2. Semântico**
- Embeddings densos congelados `BAAI/bge-small-en-v1.5` (SentenceTransformer) + Regressão Logística. O encoder permanece congelado para isolar o efeito do espaço de embeddings (sem fine-tuning end-to-end).

**3. Estrutural — IBVS (Instruction Boundary Violation Score)**
Principal contribuição metodológica. Modela prompt injection como **tentativa de violar a hierarquia de instrução** (não apenas anomalia lexical/semântica).

- **v1:** escore escalar único derivado de regras heurísticas ponderadas.
- **v2:** decompõe a evidência estrutural em componentes interpretáveis + interações + supressores + tripwire de alta especificidade.

O prompt é normalizado (NFKC, lowercase, remoção de caracteres zero-width, compactação de whitespace) e casado contra uma biblioteca fixa de regras. Saída tripla: **escore escalar**, **vetor de ativações por componente** e **flag `tripwire_alert`**.

**Pesos e lógica de disparo (IBVS v2):**

| Componente | Peso | Lógica de disparo |
|---|---|---|
| `hierarchy_override` | 3.6 | Tentativas explícitas de ignorar/esquecer/derrubar/substituir instruções, regras, políticas, guardrails (EN + DE) |
| `system_spoof` | 3.1 | Marcadores de impersonação de canal privilegiado (`system:`, `developer:`, `assistant:`, `tool:`, formas em brackets/tags) |
| `role_redefine` | 1.2 | Reatribuição de papel ("you are now", "act as", "pretend to be", equivalentes em alemão) |
| `tool_directive` | 1.8 | Instruções de uso arriscado de ferramentas/funções/plugins/APIs/shell/exfiltração (genéricas só quando ancoradas) |
| `procedural` | 0.4 | Frases procedurais (step-by-step, sequências ordenadas) — só contadas quando ancoradas por fronteira ou harm+evasion; máx. 2 hits |
| `harm_domain` | 0.9 | Termos de domínio nocivo (malware, roubo de credenciais, exploits, armas, etc.) |
| `evasion` | 2.8 | Encoding, ofuscação, leetspeak, split de caracteres, evasão de filtro, payloads base64/hex |
| `interaction_hierarchy_system` | 2.4 | Dispara quando `hierarchy_override` e `system_spoof` coocorrem |
| `interaction_system_hierarchy_spoof_chain` | 2.6 | Marcador spoof + verbo de override em janela de 140 caracteres |
| `interaction_evasion_override` | 2.0 | Evasion coocorre com override/spoof/diretiva de ferramenta efetiva |
| `interaction_tool_system` | 1.0 | Diretiva efetiva de ferramenta + spoof + (override ou evasion) |
| `interaction_harm_evasion` | 2.2 | Evidência nociva + evidência de evasão |
| `interaction_harm_procedural` | 0.8 | Nocivo + procedural ancorado + (evasion ou ferramenta arriscada) |
| `high_specific_risk_anchor` | 1.0 | Evidência estrutural de alta especificidade presente |
| `benign_context_suppression` | −1.2 | Pesquisa/aula/tarefa/defensivo/segurança/histórico sem override/evasion/harm |
| `meta_system_discussion_suppression` | −1.0 | Discussão sobre system prompts/injeção/jailbreak/red-teaming sem override/evasion/harm/ferramenta |
| `length_penalty` | −0.4 / −0.6 | −0.4 acima de 220 tokens; −0.6 acima de 320 tokens |

**`tripwire_alert` = 1** quando: override+spoof coocorrem; spoof-chain dispara; harm+evasion com score bruto ≥ 5.0; ≥ 2 sinais maiores com score ≥ 7.0; ou score bruto ≥ 9.0.

**4. Transformers**
- RoBERTa-base e DeBERTa-base fine-tuned como sequence classifiers nos dados ID de treino — referências modernas de alta capacidade.

**5. Híbridos**
- **Roteamento semantic-centred** com shallow fusion em regiões incertas: score semântico particionado em confiante-benigno, confiante-nocivo e incerto; apenas a banda incerta é roteada para fallback expert / re-scoring (ideia de *selective prediction*).
- **Late fusion** RoBERTa + IBVS: probabilidades do RoBERTa congelado combinadas com features IBVS via Regressão Logística low-capacity. Variantes `IBVS_TOTAL` (escalar único) e `IBVS_STRUCTURED` (componentes decompostos).

### 2.6 Protocolo de treino e thresholds

- Seleção de checkpoint e threshold **separadas**.
- Encoders fine-tuned: checkpoint escolhido na validação usando macro-F1 no threshold default 0.5; threshold final selecionado pós-treino apenas com previsões de validação e **congelado** antes do teste/OOD.
- Transformer fine-tuning: AdamW, lr 2×10⁻⁵, weight decay 0.01, warmup linear (settings fixos de baseline).
- Thresholds: varredura para maximizar macro-F1 na validação; baixo FPR via curva ROC (TPR nos níveis de FPR pré-definidos).

### 2.7 Métricas

- **Ranking:** ROC-AUC, AUC-PR.
- **Deploy:** macro-F1, acurácia, e **TPR @ FPR fixo de 1%, 5%, 10%** (benignos dominam tráfego real; FPR modesta pode tornar a salvaguarda inutilizável).
- Macro-F1 usado como critério comum de seleção de threshold (equilibra precision/recall entre classes).

### 2.8 Baseline externo de salvaguarda

LLM Guard PromptInjection scanner (ProtectAI `deberta-v3-base-prompt-injection-v2`) avaliado como referência off-the-shelf — **não** fine-tuned no split ID. Incluído para testar se a dependência de regime é só das famílias internas ou também de uma salvaguarda estabelecida.

### 2.9 Ambiente de execução

macOS, Python 3.10+, Apple MacBook Pro M1 Pro, 16 GB RAM. Rerun completo de notebook ≈ 40 min (com caches de modelo/dados).

---

## 3. Contribuições

1. **IBVS (Instruction Boundary Violation Score):** framework de features estruturais interpretáveis que decompõe o risco em componentes específicos por mecanismo (hierarchy override, system spoofing, role redefinition, tool directives, harm, evasion, procedural). Funciona tanto como input de modelo (escalar ou vetor) quanto como **camada de auditoria/explicabilidade por prompt** (registra quais regras dispararam).

2. **Framework de avaliação deployment-aware multi-regime:** compara famílias lexicais, semânticas, estruturais e transformer sob splits repetidos, múltiplos regimes OOD e métricas tanto de ranking quanto thresholded a baixo FPR.

3. **Achado central — detecção é *regime-dependent*:** nenhuma família domina em todos os cenários. Transformers (RoBERTa/DeBERTa) são mais fortes em ID e OOD amplo; TF-IDF surpreendentemente lidera no benchmark hard-negative; IBVS híbridos ganham em regimes de injeção padrão mas perdem sob FPR mais estrito. A vantagem depende do tipo de evidência exposta pelo regime.

4. **Gap entre ranking e efetividade de deploy** — ligado a **calibração:** modelos com boa AUC-PR/ROC-AUC colapsam sob FPR estrito porque seus scores não estão alinhados com regiões de decisão conservadoras (over-confidence). Apontado como o principal gargalo e agenda futura.

5. **Dependência de regime no baseline externo:** LLM Guard é competitivo no benchmark de injeção padrão (0.7235 macro-F1, melhor AUC-PR/ROC-AUC/TPR@1% entre comparados), mas degrada fortemente em OOD amplo e hard-negatives (0.4537 macro-F1, TPR@1% = 0.0000). Isso reforça (não enfraquece) a interpretação regime-dependent.

6. **Sinais estruturais como complemento, não substituto:** IBVS é leitura seletiva — melhor quando ataques expressam manipulação explícita de fronteira; menos útil quando o sinal já é lexicalmente diagnóstico ou primariamente semântico.

7. **Princípios práticos:** (a) escolher o detector pelo regime de ataque esperado; (b) tratar calibração como componente padrão do pipeline (ex.: temperature scaling) antes de selecionar thresholds; (c) encarar sinais estruturais como ferramentas de interpretabilidade e análise forense, não como ganhos universais de performance.

---

## 4. Resultados principais (médias sobre Splits A–C)

| Regime | Modelo | Macro-F1 | AUC-PR | ROC-AUC | TPR@1% | TPR@5% | TPR@10% |
|---|---|---|---|---|---|---|---|
| ID | RoBERTa+IBVS Total | 0.9541 ± 0.0354 | 0.9923 | 0.9825 | 0.5633 | 0.9599 | 0.9846 |
| ID | DeBERTa-base | 0.9346 ± 0.0303 | 0.9944 | 0.9862 | 0.7442 | 0.9446 | 0.9723 |
| Primary OOD | DeBERTa-base | 0.7528 ± 0.0417 | 0.8648 | 0.8540 | 0.3368 | 0.5217 | 0.5851 |
| Primary OOD | RoBERTa+IBVS Total | 0.7497 ± 0.0297 | 0.8846 | 0.8674 | **0.4557** | 0.5877 | 0.6328 |
| HN-Inj. OOD | TF–IDF only | 0.8082 ± 0.0563 | 0.9079 | 0.9232 | 0.2320 | 0.5447 | 0.7522 |
| HN-Inj. OOD | DeBERTa-base | 0.7313 ± 0.0581 | 0.8641 | 0.8784 | **0.3520** | 0.5998 | 0.6431 |
| Std-Inj. OOD | Semantic+IBVS Boost | 0.7296 ± 0.0091 | 0.7679 | 0.8027 | 0.0625 | 0.2356 | 0.4098 |
| Std-Inj. OOD | RoBERTa+IBVS Structured | 0.6366 ± 0.0276 | 0.7956 | 0.8117 | **0.1179** | 0.3054 | 0.4757 |

**Comparação com baseline externo (LLM Guard):**

| Regime | Modelo | Macro-F1 | ROC-AUC | TPR@1% |
|---|---|---|---|---|
| ID | LLM Guard | 0.3221 ± 0.0200 | 0.8019 | 0.1109 |
| Primary OOD | LLM Guard | 0.3408 | 0.6688 | 0.0260 |
| HN-Inj. OOD | LLM Guard | 0.4537 | 0.4341 | 0.0000 |
| Std-Inj. OOD | LLM Guard | 0.7235 | 0.8258 | 0.1902 |

**Análise por slice (quartis de comprimento/complexidade lexical):**

| Comportamento representativo | Família líder | Macro-F1 | TPR@1% |
|---|---|---|---|
| Prompts curtos e de baixo contexto | DeBERTa transformer | 0.705 | 0.286 |
| Benignos com muitos triggers | TF–IDF + flags | 0.503 | 0.000 |
| Padrões de injeção explícita | Hybrid Semantic–IBVS | 0.725 | 0.171 |
| Padrões de nocividade semântica | Hybrid RoBERTa+IBVS | 0.694 | 0.520 |

---

## 5. Código

**Disponível.** O arXiv indica "Code will be released", mas a versão PMLR (UK AI Conference 2026) já aponta o repositório público:

- **Repositório:** https://github.com/TimiAkinrele/ai-jailbreak-framework
- Notebook-first, ~18 commits, sem descrição/stars.
- **Estrutura:** `data/raw/`, `data/processed/`, `notebooks/`, `src/`, `experiments/results/`, `tests/`, `scripts/`.
- **Módulos-chave:**
  - `src/features/ibvs.py` — implementação do IBVS
  - `src/data/external_ood_loaders.py` — loaders OOD externos (r1char9, NotInject)
  - `src/evaluation/eval_metrics.py`
  - `src/models/transformer_baselines.py`, `src/models/transformer_ibvs_hybrid.py`
- **Notebooks (02–08):** preprocessing, feature engineering, ablation study, semantic baseline, hybrid routing, transformer baselines, RoBERTa+IBVS hybrid.
- **Dados:** arquivos brutos rastreados + benchmark processado (verificação de consistência dos resultados ID/OOD).
- **Execução:** Python 3.10+, `pip install -r requirements.txt`, `pytest` para verificação, notebooks por split via `SPLIT_TAG=A/B/C`; diagnóstico completo com `WRITE_MINIMAL_OUTPUTS=0`. Rerun completo ≈ 40 min em M1 Pro.

---

## 6. Limitações e trabalho futuro (dos autores)

- Ausência de estudo dedicado de calibração (reliability diagrams / expected calibration error).
- Não foram incluídas intervenções de calibração — o estudo caracteriza o gap, mas não o resolve.
- Trabalho futuro: thresholding calibration-aware, streams de prompt mais realistas, e verificar se o desempenho deployment-aware se mantém quando o comportamento adversarial evolui além dos benchmarks públicos fixos.

---

## 7. Referências-chave citadas

- Greshake et al. (2023) — indirect prompt injection em aplicações reais.
- Yi et al. (2025) — BIPIA (indirect prompt injection em LLMs).
- Wallace et al. (2024) — The instruction hierarchy.
- Ye et al. (2026) — Prompt injection como role confusion.
- Guo et al. (2017) — On calibration of modern neural networks.
- Geifman & El-Yaniv (2017) — Selective classification.
- Li & Liu (2024) — InjecGuard / over-defense (benchmark NotInject).
- Ji et al. (2025) — detecção com modelo pré-treinado + feature engineering heurística.
- Liu et al. (2025) — DataSentinel (framework game-theoretic).
- Sharma et al. (2025) — Constitutional classifiers.
