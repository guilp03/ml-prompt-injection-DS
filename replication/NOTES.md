# Notas de adaptação

- Repo de referência: `TimiAkinrele/ai-jailbreak-framework` (commit `ec5cac98e942aa38c49018a364e5cb2a9ae138e1`),
  copiado verbatim para a raiz deste repo como base da reprodução.
- Nosso tooling vive só em `replication/`; a árvore do ref não é editada.
- `experiments/results/` é o **oráculo** (tabelas commitadas pelos autores) e
  nunca é sobrescrito: cada execução roda numa cópia isolada em
  `replication/runs/<run_id>/`.
- Escopo local (Fase 1): notebooks `04` (lexical+IBVS), `05` (semântico BGE) e
  `06` (híbridos semânticos). Transformers (`07`,`08`) ficam para a Fase 2
  (cluster APUANA); smoke local é opcional e lento (configs `num_epochs=3`,
  `max_length=256` são fixas no código).
- Rodamos com `WRITE_MINIMAL_OUTPUTS=0` porque os regimes OOD hard-negative e
  std-injection (necessários à conclusão qualitativa central) só são gravados
  nesse modo.
- torch instalado como build **CPU** (máquina local sem GPU).
- Ambiente: Python 3.10.22, torch 2.14.1+cpu, transformers 5.20.0,
  sentence-transformers 6.1.0. `pytest` do ref: 31 passed.
- Dependências NÃO pinadas em versões (paridade com o `requirements.txt` do ref).
- `build_run_id` usa `time.gmtime` (não `localtime`) para ser independente de fuso.
- Desvios adicionais: registrar aqui diferenças de versão de libs, falhas de
  teste do ref, e qualquer divergência numérica relevante.
