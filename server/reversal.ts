import { propagateBeliefs, validateOptions } from './propagation.js';
import type {
  GraphDocument,
  GraphEdge,
  PropagateOptions,
  ReversalResult,
  ReversalSide,
  SensitivityPointStatus
} from '../shared/graph.js';

export const MIN_REVERSAL_EDGES = 1;
export const MAX_REVERSAL_EDGES = 4;
/** 候选权重为 0、0.1…1.0，共 11 档。 */
export const REVERSAL_WEIGHT_STEPS = 11;
const DECISION_THRESHOLD = 0.5;
/** L1 改变量比较容差：候选值在 0.1 网格上，与当前权重（0.05 步进）的差值为 0.05 的整数倍。 */
const L1_TOLERANCE = 1e-9;

/**
 * 校验结论翻转方案的选择：目标必须是观点（claim），可调连线 1–4 条且均存在。
 * 图本身的严格校验由 validateGraphDocument 在调用处完成。
 */
export function validateReversalSelection(
  graph: GraphDocument,
  claimNodeId: unknown,
  desiredSide: unknown,
  edgeIds: unknown
): { claimId: string; side: ReversalSide; edges: GraphEdge[] } {
  if (typeof claimNodeId !== 'string' || !claimNodeId.trim()) {
    throw new Error('缺少目标观点 claimNodeId。');
  }
  const claim = graph.nodes.find((node) => node.id === claimNodeId);
  if (!claim) throw new Error(`目标观点「${claimNodeId}」不存在，无法生成方案。`);
  if (claim.kind !== 'claim') {
    throw new Error(`目标节点「${claim.label}」必须是观点（claim）节点。`);
  }
  if (desiredSide !== 'atLeastHalf' && desiredSide !== 'belowHalf') {
    throw new Error('期望侧 desiredSide 必须是 atLeastHalf（≥0.5）或 belowHalf（<0.5）。');
  }
  if (!Array.isArray(edgeIds) || edgeIds.length < MIN_REVERSAL_EDGES) {
    throw new Error(`至少选择 ${MIN_REVERSAL_EDGES} 条可调连线。`);
  }
  if (edgeIds.length > MAX_REVERSAL_EDGES) {
    throw new Error(`可调连线最多选择 ${MAX_REVERSAL_EDGES} 条（候选组合数为 11^n）。`);
  }
  const seen = new Set<string>();
  const edges: GraphEdge[] = edgeIds.map((raw) => {
    if (typeof raw !== 'string' || !raw.trim()) throw new Error('可调连线 id 无效。');
    const id = raw;
    if (seen.has(id)) throw new Error(`可调连线「${id}」被重复选择。`);
    seen.add(id);
    const edge = graph.edges.find((item) => item.id === id);
    if (!edge) throw new Error(`可调连线「${id}」不存在，无法调整。`);
    return edge;
  });
  // 权重向量的字典序坐标按连线 id 升序排列，保证结果与选择顺序无关、稳定可复现。
  edges.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  return { claimId: claim.id, side: desiredSide, edges };
}

const satisfiesSide = (value: number, side: ReversalSide): boolean =>
  side === 'atLeastHalf'
    ? value >= DECISION_THRESHOLD - L1_TOLERANCE
    : value < DECISION_THRESHOLD - L1_TOLERANCE;

/**
 * 结论翻转方案：把 1–4 条可调连线的边权限定在 0、0.1…1.0 共 11 个候选值，
 * 沿用当前传播参数穷举全部 11^n 个组合（在内存的图副本上重算传播，不保存图）。
 * 只接受收敛且目标观点终值落在期望侧的组合，并按以下顺序稳定选出全局最优：
 *   1. 相对当前权重的 L1 改变量最小；
 *   2. 改动边数最少；
 *   3. 按连线 id 排序后的权重向量字典序最小。
 * 当前图已满足期望时直接返回零改动方案（只评估当前权重一次）。
 */
export function planConclusionReversal(
  graph: GraphDocument,
  claimNodeId: string,
  desiredSide: ReversalSide,
  rawEdgeIds: unknown,
  suppliedOptions?: PropagateOptions
): ReversalResult {
  const options = validateOptions(suppliedOptions);
  const { claimId, side, edges } = validateReversalSelection(graph, claimNodeId, desiredSide, rawEdgeIds);
  const orderedEdgeIds = edges.map((edge) => edge.id);
  // 当前权重保持画布原值（可能为 0.05 网格值）：L1 改变量与“改动边”都相对它计算。
  const currentWeights = edges.map((edge) => edge.weight);
  const candidates = Array.from({ length: REVERSAL_WEIGHT_STEPS }, (_, index) => index / 10);

  const runWithWeights = (weights: number[]) => {
    const weightByEdgeId = new Map(orderedEdgeIds.map((id, index) => [id, weights[index]]));
    const candidateGraph: GraphDocument = {
      ...graph,
      edges: graph.edges.map((edge) =>
        weightByEdgeId.has(edge.id) ? { ...edge, weight: weightByEdgeId.get(edge.id)! } : edge
      )
    };
    return propagateBeliefs(candidateGraph, options);
  };

  // 先用画布当前权重评估一次：当前图已收敛且目标观点在期望侧时，零改动直接返回。
  const baseline = runWithWeights(currentWeights);
  const baselineStatus: SensitivityPointStatus = baseline.converged
    ? 'converged'
    : baseline.oscillating
      ? 'oscillating'
      : 'limit';

  if (baseline.converged && satisfiesSide(baseline.finalBeliefs[claimId], side)) {
    return {
      claimNodeId: claimId,
      desiredSide: side,
      edgeIds: orderedEdgeIds,
      candidateSteps: REVERSAL_WEIGHT_STEPS,
      evaluatedCount: 1,
      baselineStatus,
      baselineValue: baseline.finalBeliefs[claimId],
      alreadySatisfied: true,
      feasibleCount: 1,
      feasible: true,
      best: {
        suggestions: edges.map((edge) => ({
          edgeId: edge.id,
          sourceId: edge.source,
          targetId: edge.target,
          stance: edge.stance,
          originalWeight: edge.weight,
          suggestedWeight: edge.weight,
          changed: false
        })),
        targetValue: baseline.finalBeliefs[claimId],
        l1Change: 0,
        changedEdgeCount: 0,
        iterations: baseline.iterations
      },
      oscillatingCount: 0,
      limitCount: 0,
      infeasibleCount: 0,
      options,
      message:
        `当前图已满足期望（目标观点收敛终值 ${baseline.finalBeliefs[claimId].toFixed(4)} ${
          side === 'atLeastHalf' ? '≥' : '<'
        } 0.5），无需改动任何连线权重。`
    };
  }

  let oscillatingCount = 0;
  let limitCount = 0;
  let infeasibleCount = 0;
  let feasibleCount = 0;
  // 最优解的三元比较键：(L1 改变量, 改动边数, 权重向量字典序)，逐级严格更小才替换。
  let bestWeights: number[] | null = null;
  let bestL1 = Infinity;
  let bestChanged = Infinity;
  let bestValue = 0;
  let bestIterations = 0;

  const trialWeights = new Array<number>(edges.length).fill(0);

  /**
   * 按连线 id 顺序（orderedEdgeIds 已按 id 升序）逐层枚举候选权重。
   * 枚举顺序天然就是权重向量的字典序：先到的最优组合在三元键完全相同时胜出，
   * 因此显式比较仅需处理同一枚举位置上的微小浮点差异，字典序兜底保证结果稳定。
   */
  const evaluate = () => {
    const run = runWithWeights(trialWeights);
    if (!run.converged) {
      if (run.oscillating) oscillatingCount += 1;
      else limitCount += 1;
      return;
    }
    const targetValue = run.finalBeliefs[claimId];
    if (!satisfiesSide(targetValue, side)) {
      infeasibleCount += 1;
      return;
    }
    feasibleCount += 1;
    let l1Change = 0;
    let changedEdgeCount = 0;
    for (let index = 0; index < trialWeights.length; index += 1) {
      const delta = Math.abs(trialWeights[index] - currentWeights[index]);
      l1Change += delta;
      if (delta > L1_TOLERANCE) changedEdgeCount += 1;
    }
    const better =
      l1Change < bestL1 - L1_TOLERANCE ||
      (Math.abs(l1Change - bestL1) <= L1_TOLERANCE &&
        (changedEdgeCount < bestChanged ||
          (changedEdgeCount === bestChanged && isLexicographicallySmaller(trialWeights, bestWeights))));
    if (better) {
      bestWeights = [...trialWeights];
      bestL1 = l1Change;
      bestChanged = changedEdgeCount;
      bestValue = targetValue;
      bestIterations = run.iterations;
    }
  };

  const enumerate = (depth: number) => {
    if (depth === edges.length) {
      evaluate();
      return;
    }
    for (const candidate of candidates) {
      trialWeights[depth] = candidate;
      enumerate(depth + 1);
    }
  };
  enumerate(0);

  const evaluatedCount = REVERSAL_WEIGHT_STEPS ** edges.length;

  if (bestWeights === null) {
    const sideText = side === 'atLeastHalf' ? '≥ 0.5' : '< 0.5';
    return {
      claimNodeId: claimId,
      desiredSide: side,
      edgeIds: orderedEdgeIds,
      candidateSteps: REVERSAL_WEIGHT_STEPS,
      evaluatedCount,
      baselineStatus,
      baselineValue: baseline.converged ? baseline.finalBeliefs[claimId] : null,
      alreadySatisfied: false,
      feasibleCount: 0,
      feasible: false,
      best: null,
      oscillatingCount,
      limitCount,
      infeasibleCount,
      options,
      message:
        `已穷举 ${evaluatedCount} 个权重组合（${edges.length} 条可调连线 × 11 档候选值 0–1）：` +
        `收敛但终值未达期望（${sideText}）${infeasibleCount} 个、振荡 ${oscillatingCount} 个、` +
        `达到上限 ${limitCount} 个，不存在可行的结论翻转方案。`
    };
  }

  return {
    claimNodeId: claimId,
    desiredSide: side,
    edgeIds: orderedEdgeIds,
    candidateSteps: REVERSAL_WEIGHT_STEPS,
    evaluatedCount,
    baselineStatus,
    baselineValue: baseline.converged ? baseline.finalBeliefs[claimId] : null,
    alreadySatisfied: false,
    feasibleCount,
    feasible: true,
    best: {
      suggestions: edges.map((edge, index) => ({
        edgeId: edge.id,
        sourceId: edge.source,
        targetId: edge.target,
        stance: edge.stance,
        originalWeight: edge.weight,
        suggestedWeight: bestWeights![index],
        changed: Math.abs(bestWeights![index] - currentWeights[index]) > L1_TOLERANCE
      })),
      targetValue: bestValue,
      l1Change: bestL1,
      changedEdgeCount: bestChanged,
      iterations: bestIterations
    },
    oscillatingCount,
    limitCount,
    infeasibleCount,
    options,
    message:
      `已穷举 ${evaluatedCount} 个权重组合：可行 ${feasibleCount} 个、振荡 ${oscillatingCount} 个、` +
      `达到上限 ${limitCount} 个。最优方案改动 ${bestChanged} 条连线（L1 改变量 ${bestL1.toFixed(1)}），` +
      `目标观点收敛终值 ${bestValue.toFixed(4)}（${side === 'atLeastHalf' ? '≥' : '<'} 0.5）。`
  };
}

/** 权重向量字典序比较：a < b（按 0.1 网格档位逐坐标比较）。 */
function isLexicographicallySmaller(a: number[], b: number[] | null): boolean {
  if (b === null) return true;
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) {
    const stepA = Math.round(a[index] * 10);
    const stepB = Math.round(b[index] * 10);
    if (stepA !== stepB) return stepA < stepB;
  }
  return false;
}
