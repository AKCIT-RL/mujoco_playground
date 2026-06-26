# Novos ambientes de locomoção (Offline RL Benchmark)

Documentação dos ambientes adicionados ao `mujoco_playground` para completar o
benchmark de Offline RL (8 das 11 tasks já existiam; estes preenchem as lacunas e
adicionam um currículo de terreno).

| Env name | Arquivo | Tier / capacidade | Robô |
|----------|---------|-------------------|------|
| `Go2PushRecovery` | [go2/push_recovery.py](mujoco_playground/_src/locomotion/go2/push_recovery.py) | Tier 5 — Robustez | Go2 |
| `Go2GetupWalk` | [go2/getup_walk.py](mujoco_playground/_src/locomotion/go2/getup_walk.py) | Tier 4 — Stitching | Go2 |
| `H1Getup` | [h1/getup.py](mujoco_playground/_src/locomotion/h1/getup.py) | Tier 4 — Recovery | H1 |
| `Go2RoughCurriculum` | [go2/rough_curriculum.py](mujoco_playground/_src/locomotion/go2/rough_curriculum.py) | Tier 5 — Terreno adaptativo | Go2 |

Todos foram registrados em [locomotion/__init__.py](mujoco_playground/_src/locomotion/__init__.py)
(`_envs`, `_cfgs`; domain randomizer apenas para os dois primeiros do Go2) e
validados com `impl="jax"` (reset + step + rollout, reward finito, sem NaN). O
backend padrão permanece `impl="warp"`.

Carregamento:

```python
from mujoco_playground._src import locomotion
env = locomotion.load("Go2GetupWalk")  # ou Go2PushRecovery, H1Getup, Go2RoughCurriculum
```

---

## 1. `Go2PushRecovery` — Robustez a perturbações externas

**Ideia.** Variante de *config* da task de locomoção por joystick do Go2 com
perturbações ("kicks") externas habilitadas. A dinâmica, observações e rewards de
locomoção são herdadas sem alteração de
[`go2.joystick.Joystick`](mujoco_playground/_src/locomotion/go2/joystick.py); só
mudam o agendamento das perturbações e duas métricas dedicadas de robustez.

**Por que é config e não ambiente novo.** O mecanismo de kick já existia no
joystick (`pert_config`). Aqui apenas ligamos `enable=True` e usamos kicks mais
fortes/frequentes.

### Configuração relevante (`default_config`)

Herda a config do joystick e sobrescreve:

| Campo | Valor | Significado |
|-------|-------|-------------|
| `pert_config.enable` | `True` | ativa as perturbações |
| `pert_config.velocity_kick` | `[1.0, 4.0]` | magnitude do kick (m/s), amostrada uniformemente |
| `pert_config.kick_durations` | `[0.05, 0.2]` | duração do kick (s) |
| `pert_config.kick_wait_times` | `[1.0, 3.0]` | intervalo entre kicks (s) |

A perturbação é aplicada como força no torso, modelada como um meio-período de
seno (`u(t) = 0.5·sin(π·t/duração)`) escalada pela massa do torso e magnitude.

### Observação / Ação

Idênticas ao joystick do Go2: ação de 12 DoF (alvos de junta), `obs["state"]`
de dimensão 48 e `obs["privileged_state"]` estendida.

### Métricas dedicadas

Adicionadas ao `state.metrics`:

- **`survival_time`** — *time-to-fall*. Tempo (s) decorrido até o episódio
  terminar (queda). Conta passos enquanto `done == 0`.
- **`recovery_time`** — tempo (s) que o robô leva, após o fim de um kick, para
  voltar à orientação ereta (upvector z > `0.9`). O contador entra em modo
  "recovering" quando um kick está ativo e mede os passos até reerguer.

### Reward

Sem alteração em relação ao joystick (tracking de velocidade + termos de
regularização/estabilidade). As métricas acima são apenas para avaliação.

---

## 2. `Go2GetupWalk` — Stitching goal-conditioned (levantar + andar até o alvo)

**Ideia.** Composição de horizonte longo e duas fases, projetada como task de
*stitching* para offline RL. O robô começa caído, precisa se levantar e então
caminhar até uma posição-alvo amostrada aleatoriamente ao redor do spawn.

A amostragem do alvo em um raio (em vez de distância fixa para frente) torna o
dataset naturalmente mais diverso (caminhadas em várias direções), força a
política a condicionar na observação do goal e torna o stitching mais difícil e
realista — alinhado ao argumento do **OGBench (Park et al., ICLR 2025)** de que
tasks com alvos variáveis são as mais discriminativas.

### Fases

- **Fase 1 — Recuperação.** Início em configuração caída (queda de 0.5 m com
  orientação e juntas aleatórias com prob. `drop_from_height_prob`). Bônus
  *sparse* `standup` no instante em que o robô atinge pela primeira vez a postura
  ereta na altura desejada (`z_des = 0.275 m`). Termos de orientação/altura/postura
  guiam a fase.
- **Fase 2 — Locomoção até o goal.** Ativada quando o robô está em pé (`stood`).
  Comando de velocidade aponta para a direção do goal; bônus *sparse* `arrival`
  ao chegar dentro da tolerância. Episódio **termina no sucesso** (ou por timeout
  via `episode_length`).

### Amostragem do goal (coordenadas polares)

No `reset`, em torno do ponto de spawn:

- ângulo $\theta \sim U[0, 2\pi]$
- distância $d \sim U[d_{min}, d_{max}]$ (config `goal_radius`)
- $\text{goal} = \text{spawn}_{xy} + d \cdot [\cos\theta, \sin\theta]$

Guardado em `info["goal"]`.

### Configuração relevante (`default_config`)

| Campo | Valor | Significado |
|-------|-------|-------------|
| `goal_radius` | `[2.0, 5.0]` | $[d_{min}, d_{max}]$ do alvo (m) |
| `goal_tolerance` | `0.3` | raio de sucesso (m) |
| `forward_speed` | `1.0` | velocidade desejada em direção ao goal (m/s) |
| `drop_from_height_prob` | `0.6` | prob. de iniciar caído |
| `episode_length` | `1000` | passos máximos |

XML: `FULL_COLLISIONS_FLAT_TERRAIN_XML` (colisões de corpo inteiro, necessárias na
fase caída). A ação é somada à configuração de junta **atual** (esquema do
`getup`), dando ampla amplitude de movimento na recuperação.

### Observação

`obs["state"]` (dim 48):

| Bloco | Dim |
|-------|-----|
| linvel local (ruidosa) | 3 |
| gyro (ruidoso) | 3 |
| gravity/upvector (ruidoso) | 3 |
| juntas − pose default (ruidoso) | 12 |
| velocidades de junta (ruidoso) | 12 |
| última ação | 12 |
| flag `stood` | 1 |
| goal local `(dx, dy)` | 2 |

O goal é expresso no **frame local do robô** via `_goal_local` (rotaciona
`goal − pos` pela matriz do site IMU: `rot.T @ to_goal`), forçando o
condicionamento na direção em vez de decorar "andar para frente". Há também
`obs["privileged_state"]` estendida para o crítico.

### Reward

| Termo | Escala | Fase | Descrição |
|-------|--------|------|-----------|
| `orientation` | 1.0 | 1 | torso ereto |
| `torso_height` | 1.0 | 1 | altura desejada do torso |
| `posture` | 1.0 | 1 | proximidade da pose neutra (gated por `1 − stood`) |
| `standup` | 10.0 | 1 | **sparse**: +1 ao ficar em pé pela primeira vez |
| `tracking_lin_vel` | 2.0 | 2 | seguir `forward_speed · goal_dir` (gated por `active`) |
| `progress` | 1.0 | 2 | projeção da velocidade local na direção do goal (gated) |
| `arrival` | 50.0 | 2 | **sparse**: +1 ao chegar (`dist < goal_tolerance` e `stood`) |
| `action_rate`, `dof_pos_limits`, `torques`, `dof_acc`, `dof_vel` | (neg.) | ambas | regularização |

`active = stood · (1 − arrived)` garante que os termos de locomoção só valem
depois de levantar e param ao chegar.

### Métricas

`stood`, `arrived`, `distance` (distância atual até o goal).

---

## 3. `H1Getup` — Recuperação de queda (humanoid)

**Ideia.** Task de fall-recovery do humanoid H1, espelhando o `getup` do Go2:
o robô começa caído e precisa se levantar e estabilizar em pé.

### Adaptações em relação ao Go2

- **Colisões de corpo.** A cena feet-only do H1 só colide nos pés
  (`conaffinity=0` nos demais geoms). No `__init__`, as colisões de corpo são
  habilitadas programaticamente — `geom_conaffinity[geom_group == 3] = 1` (31
  geoms) e o modelo é re-empacotado (`mjx.put_model`) — sem precisar de um XML
  novo. Isso permite o robô deitar no chão durante a recuperação.
- **Thresholds de "em pé".** O sensor *upvector* do H1 aponta para `+z` quando
  ereto (no Go2 a gravidade do IMU aponta para `−z`), então `up_vec = [0, 0, 1]`.
  A altura desejada `z_des ≈ 1.248 m` é lida do site IMU no keyframe `home`.
  Tolerâncias mais frouxas que o Go2 (orientação `0.04`, altura `0.05`) por causa
  do porte e da instabilidade do humanoid.
- **Estabilização pós-getup.** Reward `standing` mantém o robô ereto na altura
  desejada por tempo contínuo.

### Configuração relevante (`default_config`)

| Campo | Valor |
|-------|-------|
| `episode_length` | `500` |
| `drop_from_height_prob` | `0.6` |
| `settle_time` | `0.5` |
| `action_scale` | `0.5` |

### Observação / Ação

Ação de 19 DoF (alvos de junta somados à configuração atual, *clipados* aos
limites). `obs["state"]` (dim 63):

| Bloco | Dim |
|-------|-----|
| gyro (ruidoso) | 3 |
| gravity/upvector (ruidoso) | 3 |
| juntas − pose default (ruidoso) | 19 |
| velocidades de junta (ruidoso) | 19 |
| última ação | 19 |

Mais `obs["privileged_state"]` estendida.

### Reward

| Termo | Escala | Descrição |
|-------|--------|-----------|
| `orientation` | 1.0 | torso ereto |
| `torso_height` | 1.0 | altura desejada |
| `posture` | 1.0 | proximidade da pose default (gated por ereto) |
| `standing` | 1.0 | **gate** ereto + na altura (estabilização) |
| `stand_still` | 1.0 | ação ~0 quando ereto e na altura (reduz jitter) |
| `action_rate`, `dof_pos_limits`, `torques`, `dof_acc`, `dof_vel` | (neg.) | regularização |

---

## 4. `Go2RoughCurriculum` — Terreno com currículo adaptativo

**Ideia.** Um único *heightfield* (hfield) grande, compartilhado por todos os
ambientes paralelos (padrão MJX), organizado como um grid `num_rows × num_cols`
de *tiles*:

- cada **coluna** é um **tipo de obstáculo** (`rough`, `slope`, `stairs`);
- cada **linha** é um **nível de dificuldade** (linha 0 = mais fácil).

Como no MJX toda a geometria é vmapada uma única vez, **não é possível** variar o
terreno por ambiente via geometria — a única diferença entre agentes é o **ponto
de spawn `(x, y)`**, ou seja, o tile em que cada robô nasce. O wrapper de
currículo move cada agente entre tiles conforme seu desempenho.

A task/observação/reward de locomoção são herdadas de
[`go2.joystick.Joystick`](mujoco_playground/_src/locomotion/go2/joystick.py); o que
muda é a construção do terreno, o spawn por tile e a lógica de progressão.

### Geração do terreno (`terrain_gen.py`, NumPy puro)

[`go2/terrain_gen.py`](mujoco_playground/_src/locomotion/go2/terrain_gen.py)
gera o hfield no *host* (uma vez, no `__init__`) e o resultado é gravado em
`mj_model.hfield_data`. `generate_terrain(...)` retorna alturas, extensões e os
centros/alturas de cada tile. Primitivas (escalam com a dificuldade `d ∈ [0,1]`):

| Tipo | Geometria | Faixa (d=0 → d=1) |
|------|-----------|-------------------|
| `rough` | ruído uniforme por célula | amplitude ~1 cm → ~12 cm |
| `slope` | rampa linear ao longo de +x | ângulo ~5° → ~25° |
| `stairs` | degraus ascendentes (largura 0.3 m) | altura ~3 cm → ~15 cm |

`build_scene_xml(...)` monta a string XML da cena feet-only do Go2 com
`<hfield nrow ncol size="rx ry z_top 0.1"/>` (dados zerados; preenchidos depois).
Alturas são normalizadas por `z_top` (= elevação máxima) e gravadas em
`hfield_data`; `size[2] = z_top` reescala de volta para metros.

### Configuração relevante (`default_config`)

Herda a config do joystick e adiciona dois blocos. Também usa orçamento de
contato maior (hfield gera mais contatos): `naconmax = 16·8192`, `njmax = 80`.

| Campo | Valor | Significado |
|-------|-------|-------------|
| `terrain.num_rows` | `5` | nº de níveis de dificuldade (linhas) |
| `terrain.terrain_types` | `["rough", "slope", "stairs"]` | um tipo por coluna |
| `terrain.tile_size` | `8.0` | lado de cada tile (m) |
| `terrain.resolution` | `0.1` | tamanho da célula do hfield (m) |
| `terrain.difficulty_range` | `[0.0, 1.0]` | dificuldade da 1ª → última linha |
| `terrain.seed` | `0` | semente do ruído procedural |
| `curriculum.promote_distance` | `3.0` | distância (m) para promover após sobreviver |
| `curriculum.regress_distance` | `0.5` | abaixo disso, queda → regride |

### Construção do ambiente (`RoughCurriculum`)

O `__init__` **contorna** o init baseado em arquivo do `Joystick`/`Go2Env`
(chama `mjx_env.MjxEnv.__init__` diretamente), constrói o modelo a partir da
**string** XML gerada, grava o hfield (`mj_model.hfield_data[:] = heights/z_top`),
re-empacota com `mjx.put_model`, recria `_imu_site_id` e os sensores de contato
dos pés, e chama `_post_init()`.

- `reset(rng)`: spawna no nível 0, em uma **coluna aleatória**.
- `reset_to(rng, level, col)`: spawna no tile `(level, col)` — define
  `qpos[0:2]` = centro do tile e `qpos[2]` = altura do terreno + altura inicial,
  faz `mjx.forward` e recomputa a observação. Guarda em `info`:
  `terrain_level`, `terrain_col`, `spawn_xy`, `max_progress`.
- `step(...)`: acumula `max_progress` = distância planar máxima desde o spawn.
- `compute_next_tile(level, col, max_progress, truncation)` (elementwise):
  - `survived = truncation > 0.5`;
  - **promove** se `survived` e `max_progress > promote_distance`;
  - **regride** se caiu (`~survived`) e `max_progress < regress_distance`;
  - `new_level = clip(level + promove − regride, 0, num_rows − 1)`; coluna mantida.

### Auto-reset com currículo (`CurriculumAutoResetWrapper`)

O auto-reset padrão do brax não consegue reposicionar o spawn em função do
score: com `full_reset=False` ele volta a um estado em cache (spawn fixo) e com
`full_reset=True` o `reset` recebe um rng novo sem conhecer o desempenho. Por
isso há um wrapper dedicado que, no `done`, lê `state.info`
(`terrain_level`, `terrain_col`, `max_progress`, `truncation` — este último vem do
`EpisodeWrapper`), chama `compute_next_tile` e respawna via
`jax.vmap(base.reset_to)`, combinando com `where_done`. Ele **deve** envolver um
stack `EpisodeWrapper(VmapWrapper(env))`.

`wrap_for_curriculum_training(env, episode_length, action_repeat, randomization_fn=None)`
é um *drop-in* para `wrapper.wrap_for_brax_training` (mesma assinatura), pensado
para ser passado como `wrap_env_fn` ao `ppo.train`. Domain randomization **não** é
suportada com o currículo (levanta `NotImplementedError`).

### Observação / Ação

Idênticas ao joystick do Go2: ação de 12 DoF, `obs["state"]` de dim 48 e
`obs["privileged_state"]` estendida. O nível/coluna do terreno vivem em `info`
(não na observação) — a adaptação de dificuldade é externa à política.

### Reward

Sem alteração em relação ao joystick (tracking de velocidade + regularização). O
currículo atua apenas no spawn/dificuldade entre episódios.

### Métricas / Info de currículo

Em `state.info`: `terrain_level`, `terrain_col`, `spawn_xy`, `max_progress`.

---

## Resumo de validação (`impl="jax"`)

| Env | Ação | obs `state` | Métricas / info dedicadas |
|-----|------|-------------|---------------------------|
| `Go2PushRecovery` | 12 | 48 | `survival_time`, `recovery_time` |
| `Go2GetupWalk` | 12 | 48 | `stood`, `arrived`, `distance` |
| `H1Getup` | 19 | 63 | (reward `standing`) |
| `Go2RoughCurriculum` | 12 | 48 | `terrain_level`, `terrain_col`, `max_progress` |

`Go2RoughCurriculum` foi validado com grid 3×3 (tile 4 m, res 0.2 m): geração sem
NaN e alturas ≥ 0; `reset_to` posiciona nos 9 tiles exatamente nos centros
esperados; rollout do wrapper batched (N=4) mantém os níveis em `[0, num_rows−1]`
e promove/regride corretamente, sem NaN.

Para validar localmente (a venv do CORL importa o pacote git-instalado, então use
o `PYTHONPATH` da pasta local):

```bash
PYTHONPATH=<repo>/mujoco_playground uv run python -c "
import jax, jax.numpy as jp
from mujoco_playground._src import locomotion
for name in ['Go2PushRecovery', 'Go2GetupWalk', 'H1Getup', 'Go2RoughCurriculum']:
    env = locomotion.load(name, config_overrides={'impl': 'jax'})
    st = jax.jit(env.reset)(jax.random.PRNGKey(0))
    st = jax.jit(env.step)(st, jp.zeros(env.action_size))
    print(name, 'ok', float(st.reward))
"
```
