import { compilePropagation, runCompiledPropagation, validateOptions } from './propagation.js';
export const MIN_REVERSAL_EDGES = 1;
export const MAX_REVERSAL_EDGES = 4;
/** 候选权重固定为 0、0.1、…、1.0，共 11 档。 */
export const REVERSAL_WEIGHT_STEPS = 10;
export const REVERSAL_DECISION_THRESHOLD = 0.5;
/** 建议值与原值相差不超过该容差视为“未改动”（候选值由整数比例生成，正常不会落在容差边界）。 */
const CHANGE_EPSILON = 1e-9;
/**
 * 校验结论翻转选择：目标必须是观点（claim），可调连线 1–4 条且均存在、不重复。
 * 图本身的严格校验由 validateGraphDocument 在调用处完成。
 * 返回按连线 id 升序排列的连线（穷举顺序与权重向量字典序都以此为准）。
 */
export function validateReversalSelection(graph, claimNodeId, desiredSide, edgeIds) {
    if (typeof claimNodeId !== 'string' || !claimNodeId.trim()) {
        throw new Error('缺少目标观点 claimNodeId。');
    }
    const claim = graph.nodes.find((node) => node.id === claimNodeId);
    if (!claim)
        throw new Error(`目标观点「${claimNodeId}」不存在，无法制定翻转方案。`);
    if (claim.kind !== 'claim') {
        throw new Error(`目标节点「${claim.label}」必须是观点（claim）节点。`);
    }
    if (desiredSide !== 'above' && desiredSide !== 'below') {
        throw new Error('期望侧 desiredSide 必须是 above（最终可信度 ≥ 0.5）或 below（< 0.5）。');
    }
    if (!Array.isArray(edgeIds) || edgeIds.length < MIN_REVERSAL_EDGES) {
        throw new Error(`至少选择 ${MIN_REVERSAL_EDGES} 条可调连线。`);
    }
    if (edgeIds.length > MAX_REVERSAL_EDGES) {
        throw new Error(`可调连线最多选择 ${MAX_REVERSAL_EDGES} 条（组合数为 11^n）。`);
    }
    const seen = new Set();
    const edges = edgeIds.map((raw) => {
        if (typeof raw !== 'string' || !raw.trim())
            throw new Error('可调连线 id 无效。');
        if (seen.has(raw))
            throw new Error(`可调连线「${raw}」被重复选择。`);
        seen.add(raw);
        const edge = graph.edges.find((item) => item.id === raw);
        if (!edge)
            throw new Error(`可调连线「${raw}」不存在，无法制定翻转方案。`);
        return edge;
    });
    // 字典序 / 穷举顺序都以连线 id 升序为唯一基准。
    edges.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    return { claimId: claim.id, side: desiredSide, edges };
}
/** 网格权重：用整数比例生成 0、0.1、…、1.0，避免 0.1 累加法带来的浮点漂移。 */
function gridWeight(index) {
    return Math.round((index / REVERSAL_WEIGHT_STEPS) * 1000) / 1000;
}
/**
 * 结论翻转方案：
 * 在所选 1–4 条连线上，把候选权重限定为 0、0.1、…、1.0（共 11 档），
 * 沿用当前传播参数穷举全部 11^n 个组合。当前图已收敛且满足期望时直接返回零改动方案；
 * 否则只接受收敛组合，按
 *   1) 相对当前权重的 L1 改变量最小；
 *   2) 改动边数最少；
 *   3) 按连线 id 排序后的权重向量字典序最小
 * 的顺序稳定选出全局最优。振荡 / 达到上限的组合一律不接受。
 * 全程只在内存中计算，不保存图、不写 SQLite。
 */
export function planConclusionReversal(graph, claimNodeId, desiredSide, edgeIds, suppliedOptions) {
    const options = validateOptions(suppliedOptions);
    const { claimId, side, edges } = validateReversalSelection(graph, claimNodeId, desiredSide, edgeIds);
    const orderedEdgeIds = edges.map((edge) => edge.id);
    const targetIndex = graph.nodes.findIndex((node) => node.id === claimId);
    // 编译一次传播结构：可调边各占槽位 0..n-1（穷举时覆写），其余边从槽位 n 起按图顺序排布，
    // 槽位空间互不重叠，避免与“恰好排在前 n 位”的非可调边发生撞槽。
    const adjustableIds = new Set(edges.map((edge) => edge.id));
    const slotOf = new Map();
    let nextSharedSlot = edges.length;
    graph.edges.forEach((edge) => {
        if (adjustableIds.has(edge.id))
            return;
        slotOf.set(edge.id, nextSharedSlot);
        nextSharedSlot += 1;
    });
    edges.forEach((edge, slotIndex) => slotOf.set(edge.id, slotIndex));
    const compiled = compilePropagation(graph, options, slotOf);
    const satisfies = (value) => side === 'above' ? value >= REVERSAL_DECISION_THRESHOLD : value < REVERSAL_DECISION_THRESHOLD;
    // 当前图基线：槽位 0..n-1 先放入各可调边的原权重，评估一次当前画布。
    const currentWeights = edges.map((edge) => edge.weight);
    edges.forEach((edge, slotIndex) => {
        compiled.edgeWeights[slotIndex] = edge.weight;
    });
    const baseline = runCompiledPropagation(compiled);
    const baselineValue = baseline.finalBeliefs[targetIndex];
    const currentStatus = baseline.converged
        ? 'converged'
        : baseline.oscillating
            ? 'oscillating'
            : 'limit';
    const baseResult = {
        claimNodeId: claimId,
        desiredSide: side,
        edgeIds: orderedEdgeIds,
        threshold: REVERSAL_DECISION_THRESHOLD,
        options
    };
    // 当前图已收敛且满足期望：零改动即最优，直接返回，不穷举。
    if (baseline.converged && satisfies(baselineValue)) {
        const adjustments = edges.map((edge, index) => ({
            edgeId: edge.id,
            sourceId: edge.source,
            targetId: edge.target,
            stance: edge.stance,
            currentWeight: edge.weight,
            suggestedWeight: currentWeights[index],
            changed: false
        }));
        return {
            ...baseResult,
            currentTargetValue: baselineValue,
            currentStatus,
            alreadySatisfied: true,
            feasible: true,
            adjustments,
            targetFinalValue: baselineValue,
            l1Change: 0,
            changedEdgeCount: 0,
            weightVector: currentWeights.map((weight) => Math.round(weight * 1000) / 1000),
            candidateCount: 0,
            evaluatedCount: 1,
            oscillatingCount: 0,
            limitCount: 0,
            infeasibleCount: 0,
            message: `当前图已收敛且目标可信度 ${baselineValue.toFixed(4)} ${side === 'above' ? '≥' : '<'} 0.5，` +
                '已满足期望，无需调整任何连线。'
        };
    }
    // 穷举全部 11^n 个组合（含原权重组合，保证计数完整、与基线是否收敛无关）。
    const slotCount = edges.length;
    const totalCombinations = 11 ** slotCount;
    const counters = { oscillating: 0, limit: 0, infeasible: 0 };
    let best = null;
    const bestVectorIsLess = (candidate, current) => {
        for (let index = 0; index < slotCount; index += 1) {
            if (candidate[index] < current[index])
                return true;
            if (candidate[index] > current[index])
                return false;
        }
        return false;
    };
    const vector = new Array(slotCount);
    const digits = new Array(slotCount).fill(0);
    for (let combination = 0; combination < totalCombinations; combination += 1) {
        // 混合进制展开：digits[0] 对应 id 最小的连线，为字典序最高位。
        let rest = combination;
        for (let slot = slotCount - 1; slot >= 0; slot -= 1) {
            digits[slot] = rest % 11;
            rest = Math.floor(rest / 11);
        }
        for (let slot = 0; slot < slotCount; slot += 1) {
            const weight = gridWeight(digits[slot]);
            vector[slot] = weight;
            compiled.edgeWeights[slot] = weight;
        }
        const run = runCompiledPropagation(compiled);
        if (!run.converged) {
            if (run.oscillating)
                counters.oscillating += 1;
            else
                counters.limit += 1;
            continue;
        }
        const value = run.finalBeliefs[targetIndex];
        if (!satisfies(value)) {
            counters.infeasible += 1;
            continue;
        }
        let l1 = 0;
        let changed = 0;
        for (let slot = 0; slot < slotCount; slot += 1) {
            const delta = Math.abs(vector[slot] - currentWeights[slot]);
            l1 += delta;
            if (delta > CHANGE_EPSILON)
                changed += 1;
        }
        if (!best ||
            l1 < best.l1 ||
            (l1 === best.l1 && changed < best.changed) ||
            (l1 === best.l1 && changed === best.changed && bestVectorIsLess(vector, best.vector))) {
            best = { l1, changed, vector: [...vector], value };
        }
    }
    const evaluatedCount = totalCombinations;
    if (!best) {
        return {
            ...baseResult,
            currentTargetValue: baseline.converged ? baselineValue : null,
            currentStatus,
            alreadySatisfied: false,
            feasible: false,
            adjustments: null,
            targetFinalValue: null,
            l1Change: null,
            changedEdgeCount: null,
            weightVector: null,
            candidateCount: totalCombinations,
            evaluatedCount,
            oscillatingCount: counters.oscillating,
            limitCount: counters.limit,
            infeasibleCount: counters.infeasible,
            message: `已穷举全部 ${totalCombinations} 个权重组合（评估 ${evaluatedCount} 次传播）：` +
                `振荡 ${counters.oscillating} 个、达到上限 ${counters.limit} 个、收敛但未达期望 ${counters.infeasible} 个，` +
                `没有任何收敛组合能让目标可信度落到${side === 'above' ? ' ≥ 0.5' : ' < 0.5'}。`
        };
    }
    const adjustments = edges.map((edge, index) => ({
        edgeId: edge.id,
        sourceId: edge.source,
        targetId: edge.target,
        stance: edge.stance,
        currentWeight: edge.weight,
        suggestedWeight: best.vector[index],
        changed: Math.abs(best.vector[index] - currentWeights[index]) > CHANGE_EPSILON
    }));
    return {
        ...baseResult,
        currentTargetValue: baseline.converged ? baselineValue : null,
        currentStatus,
        alreadySatisfied: false,
        feasible: true,
        adjustments,
        targetFinalValue: best.value,
        l1Change: best.l1,
        changedEdgeCount: best.changed,
        weightVector: best.vector,
        candidateCount: totalCombinations,
        evaluatedCount,
        oscillatingCount: counters.oscillating,
        limitCount: counters.limit,
        infeasibleCount: counters.infeasible,
        message: `已穷举全部 ${totalCombinations} 个权重组合（评估 ${evaluatedCount} 次传播）：` +
            `振荡 ${counters.oscillating} 个、达到上限 ${counters.limit} 个、收敛但未达期望 ${counters.infeasible} 个。` +
            `最优方案改动 ${best.changed} 条连线、L1 改变量 ${best.l1.toFixed(4)}，` +
            `目标可信度 ${baseline.converged ? `${baselineValue.toFixed(4)} → ` : ''}${best.value.toFixed(4)}` +
            `（${side === 'above' ? '≥' : '<'} 0.5）。`
    };
}
