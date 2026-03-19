/**
 * promptengine.js — v1.2
 * ComfyUI-PromptEngine 前端扩展
 *
 * 修复 v1.2：
 *   - [BUG1] locale 读取：兼容新旧版 ComfyUI API，多路径兜底
 *   - [BUG2] category 联动：改用 nodeType.prototype.onConnectionsChange
 *            + widget value 监听，解决 callback 在新版不触发的问题
 *   - [BUG3] 后端已删除 control_after_generate，前端不再处理
 */

import { app } from "../../scripts/app.js";

// ─────────────────────────────────────────────
// 常量
// ─────────────────────────────────────────────

const STYLE_RANDOM = "🎲 Random Style";
const STYLE_SKIP   = "── (skip) ──";

const DIMS = [
    "ethnicity", "gender", "age_appearance", "subject_appearance",
    "hair_style", "hair_color",
    "outfit", "accessories",
    "pose", "body_direction",
    "expression", "gaze",
    "location_type", "background_props", "atmosphere",
    "shot_angle", "shot_distance", "composition",
    "lighting", "color_grade", "visual_style",
];

// ─────────────────────────────────────────────
// 词典缓存
// ─────────────────────────────────────────────

let DICT_CACHE   = null;
let DICT_LOADING = null;

async function loadDictionaries() {
    if (DICT_CACHE)   return DICT_CACHE;
    if (DICT_LOADING) return DICT_LOADING;

    DICT_LOADING = fetch("/promptengine/dictionaries")
        .then(r => r.json())
        .then(data => {
            DICT_CACHE = data;
            console.log("[PromptEngine] 词典加载完成，维度数:", Object.keys(data).length);
            return data;
        })
        .catch(err => {
            console.error("[PromptEngine] 词典加载失败:", err);
            DICT_CACHE = {};
            return {};
        });

    return DICT_LOADING;
}

// ─────────────────────────────────────────────
// Locale 读取（多路径兜底）
// ─────────────────────────────────────────────

/**
 * 兼容新旧版 ComfyUI 的 locale 读取。
 *
 * 新版 ComfyUI (2024+) 将设置迁移到 app.extensionManager.setting，
 * 旧版用 app.ui.settings.getSettingValue。
 * 两种方式都尝试，任一成功即返回。
 */
function getCurrentLang() {
    let locale = null;

    // 方式 1：新版 API（extensionManager）
    try {
        locale = app.extensionManager?.setting?.get?.("Comfy.Locale");
    } catch {}

    // 方式 2：旧版 API（ui.settings）
    if (!locale) {
        try {
            // 静默调用，忽略 deprecated 警告
            locale = app.ui?.settings?.getSettingValue?.("Comfy.Locale");
        } catch {}
    }

    // 方式 3：直接读 localStorage（ComfyUI 将设置持久化在此）
    if (!locale) {
        try {
            const stored = localStorage.getItem("Comfy.Settings");
            if (stored) {
                const parsed = JSON.parse(stored);
                locale = parsed?.["Comfy.Locale"];
            }
        } catch {}
    }

    // 方式 4：读 html lang 属性（部分版本会设置）
    if (!locale) {
        try {
            locale = document.documentElement.lang;
        } catch {}
    }

    return (locale && locale.startsWith("zh")) ? "zh" : "en";
}

// ─────────────────────────────────────────────
// 工具函数
// ─────────────────────────────────────────────

function getClusterDisplayName(cluster) {
    const lang = getCurrentLang();
    return (lang === "zh" && cluster.canonical_name_zh)
        ? cluster.canonical_name_zh
        : (cluster.canonical_name ?? "");
}

function buildStyleValues(dim, dictData) {
    const values   = [STYLE_RANDOM, STYLE_SKIP];
    
    // 特殊处理 gender 维度（硬编码）
    if (dim === "gender") {
        const lang = getCurrentLang();
        if (lang === "zh") {
            values.push("男性", "女性");
        } else {
            values.push("man", "woman");
        }
        return values;
    }
    
    const clusters = dictData?.[dim]?.clusters ?? {};
    for (const cluster of Object.values(clusters)) {
        values.push(getClusterDisplayName(cluster));
    }
    return values;
}

function buildDisplayToKeyMap(dim, dictData) {
    const map = {
        [STYLE_RANDOM]: STYLE_RANDOM,
        [STYLE_SKIP]:   STYLE_SKIP,
    };
    
    // 特殊处理 gender 维度（硬编码）
    if (dim === "gender") {
        const lang = getCurrentLang();
        if (lang === "zh") {
            map["男性"] = "man";
            map["女性"] = "woman";
        } else {
            map["man"] = "man";
            map["woman"] = "woman";
        }
        return map;
    }
    
    const clusters = dictData?.[dim]?.clusters ?? {};
    for (const [key, cluster] of Object.entries(clusters)) {
        map[getClusterDisplayName(cluster)] = key;
    }
    return map;
}

/**
 * 重建 style widget 的选项列表。
 * 切换维度时 value 退回 STYLE_RANDOM；同维度刷新时尝试保持当前选项。
 */
function refreshStyleWidget(styleWidget, dim, dictData, keepValue = false) {
    const values = buildStyleValues(dim, dictData);
    const keyMap = buildDisplayToKeyMap(dim, dictData);

    // 记录切换前选中的 key
    const currentKey = styleWidget._keyMap
        ? (styleWidget._keyMap[styleWidget.value] ?? styleWidget.value)
        : styleWidget.value;

    // 在新 keyMap 中找对应 key 的显示名
    const newDisplayForKey = keepValue
        ? Object.entries(keyMap).find(([, k]) => k === currentKey)?.[0]
        : null;

    styleWidget.options.values = values;
    styleWidget._keyMap        = keyMap;
    styleWidget.value = (newDisplayForKey && values.includes(newDisplayForKey))
        ? newDisplayForKey
        : STYLE_RANDOM;  // 切换维度或找不到对应值时，默认为 Random
    
    // 特殊处理：如果是 gender 维度且当前 value 是 key（man/woman），转换为显示名称
    if (dim === "gender" && (styleWidget.value === "man" || styleWidget.value === "woman")) {
        const lang = getCurrentLang();
        styleWidget.value = (lang === "zh") 
            ? (styleWidget.value === "man" ? "男性" : "女性")
            : styleWidget.value;
    }

    app.graph?.setDirtyCanvas(true, true);
}

function installSerializeHook(styleWidget, dim) {
    // 不再进行 key 转换，直接使用显示名称
    // 因为后端会收到显示名称，并通过显示名称查找对应的 cluster
    // 但 gender 维度需要特殊处理：将中文显示名称转换为英文 key
    styleWidget.serializeValue = async function() {
        if (dim === "gender") {
            // 将中文显示名称转换为英文 key
            if (this.value === "男性") return "man";
            if (this.value === "女性") return "woman";
        }
        return this.value;
    };
}

// ─────────────────────────────────────────────
// PromptEngine Node — 单维度节点
// ─────────────────────────────────────────────

function setupPromptEngineNode(nodeType) {
    const origOnNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = async function () {
        origOnNodeCreated?.apply(this, arguments);

        const node     = this;
        const dictData = await loadDictionaries();

        const categoryWidget = node.widgets?.find(w => w.name === "category");
        const styleWidget    = node.widgets?.find(w => w.name === "style");

        if (!categoryWidget || !styleWidget) {
            console.warn("[PromptEngine] Node: 找不到 category 或 style widget");
            return;
        }

        // 构建维度显示名称映射（用于 category widget）
        const dimDisplayMap = {};
        for (const dim of DIMS) {
            const display = dictData?.[dim]?.display;
            if (display) {
                dimDisplayMap[dim] = (getCurrentLang() === "zh") ? display.zh : display.en;
            } else {
                dimDisplayMap[dim] = dim;
            }
        }

        // 构建 category widget 的显示值映射
        const categoryDisplayToKey = {};
        const categoryValues = [];
        for (const dim of DIMS) {
            const displayName = dimDisplayMap[dim] || dim;
            categoryValues.push(displayName);
            categoryDisplayToKey[displayName] = dim;
        }

        // 保存映射关系到 widget
        categoryWidget._displayToKey = categoryDisplayToKey;
        categoryWidget._keyToDisplay = dimDisplayMap;

        // 更新 category widget 的显示选项
        categoryWidget.options.values = categoryValues;

        // 将当前 value 转换为显示名称
        const currentKey = categoryWidget.value;
        if (dimDisplayMap[currentKey]) {
            categoryWidget.value = dimDisplayMap[currentKey];
        }

        // 初始化
        refreshStyleWidget(styleWidget, categoryDisplayToKey[categoryWidget.value] ?? DIMS[0], dictData, false);
        installSerializeHook(styleWidget, categoryDisplayToKey[categoryWidget.value] ?? DIMS[0]);

        // 添加序列化钩子，将显示名称转换回 key
        categoryWidget.serializeValue = async function() {
            // 将显示名称转换为实际的 key（用于后端处理）
            const displayValue = this.value;
            const key = this._displayToKey?.[displayValue] || displayValue;
            // 注意：这里返回 key，因为 category 参数需要告诉后端是哪个维度
            return key;
        };

        // 添加 onConfigure 钩子，用于工作流加载时恢复状态
        const origOnConfigure = node.onConfigure;
        node.onConfigure = function() {
            const ret = origOnConfigure?.apply(this, arguments);
            // 工作流加载后，确保 value 是显示名称而不是 key
            const currentKey = categoryWidget._displayToKey?.[categoryWidget.value] 
                ? categoryWidget.value  // 已经是显示名称
                : categoryWidget._keyToDisplay?.[categoryWidget.value];  // 需要从 key 转换为显示名称
            
            if (currentKey && categoryWidget.options.values.includes(currentKey)) {
                categoryWidget.value = currentKey;
            }
            return ret;
        };

        // ── category 变化监听 ─────────────────────────────────────────────
        //
        // ComfyUI 新版中 widget.callback 有时不触发。
        // 最可靠的方式：在 widget 上定义一个 getter/setter 拦截 value 变化。
        //
        let _categoryValue = categoryWidget.value;
        Object.defineProperty(categoryWidget, "value", {
            get() { return _categoryValue; },
            set(newVal) {
                if (newVal !== _categoryValue) {
                    // 如果是显示名称，转换为 key
                    const newKey = categoryWidget._displayToKey?.[newVal] || newVal;
                    _categoryValue = newVal;
                    // 切换维度，重置 style
                    refreshStyleWidget(styleWidget, newKey, dictData, false);
                    // 更新 serializeHook 的维度信息
                    installSerializeHook(styleWidget, newKey);
                } else {
                    _categoryValue = newVal;
                }
            },
            configurable: true,
        });

        // 同时保留 callback 兜底（旧版 ComfyUI 走这里）
        const origCallback = categoryWidget.callback;
        categoryWidget.callback = function(value, canvas, node_, pos, e) {
            origCallback?.call(this, value, canvas, node_, pos, e);
            if (value !== styleWidget._lastCategoryValue) {
                styleWidget._lastCategoryValue = value;
                const key = categoryWidget._displayToKey?.[value] || value;
                refreshStyleWidget(styleWidget, key, dictData, false);
                // 更新 serializeHook 的维度信息
                installSerializeHook(styleWidget, key);
            }
        };
        styleWidget._lastCategoryValue = categoryWidget.value;
    };
}

// ─────────────────────────────────────────────
// PromptEngine Full — 全维度节点
// ─────────────────────────────────────────────

function setupPromptEngineFull(nodeType) {
    const origOnNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = async function () {
        origOnNodeCreated?.apply(this, arguments);

        const node     = this;
        const dictData = await loadDictionaries();
        const lang = getCurrentLang();

        for (const dim of DIMS) {
            const styleWidget = node.widgets?.find(w => w.name === `${dim}_style`);
            if (!styleWidget) {
                console.warn(`[PromptEngine] Full: 找不到 widget: ${dim}_style`);
                continue;
            }
            
            // 获取维度的显示名称
            const display = dictData?.[dim]?.display;
            const displayName = (lang === "zh" && display?.zh) ? display.zh : (display?.en || dim);
            
            // 设置 widget 的显示名称
            styleWidget.label = displayName;
            
            // 刷新 widget 选项
            refreshStyleWidget(styleWidget, dim, dictData, false);
            installSerializeHook(styleWidget, dim);
        }
    };
}

// ─────────────────────────────────────────────
// 注册扩展
// ─────────────────────────────────────────────

app.registerExtension({
    name: "ComfyUI.PromptEngine",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        loadDictionaries();

        if (nodeData.name === "PromptEngineNode") {
            setupPromptEngineNode(nodeType);
        }
        if (nodeData.name === "PromptEngineFull") {
            setupPromptEngineFull(nodeType);
        }
    },
});
