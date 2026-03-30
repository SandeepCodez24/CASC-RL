
import * as ort from 'onnxruntime-web';
import model_stats from './model_stats.json';

// Initialize session variables
let wm_session = null;
let actor_session = null;

// Initialize ONNX Sessions
export async function initAgentModels() {
    try {
        console.log("Initializing ONNX Agents...");
        wm_session = await ort.InferenceSession.create('/world_model.onnx');
        actor_session = await ort.InferenceSession.create('/mappo.onnx');
        console.log("Neural Agents READY. Distributed MAPPO online.");
        return true;
    } catch (e) {
        console.error("Model Load Error: fallback to rule-based logic.", e);
        return false;
    }
}

// Normalize state vector (8 dims)
function normalizeState(s) {
    const { mean, var: vari } = model_stats.state_normalizer;
    return s.map((val, i) => (val - mean[i]) / (Math.sqrt(vari[i]) + 1e-8));
}

// Inference Loop for a single satellite
export async function getActionFromModel(satState) {
    if (!actor_session || !wm_session) return 1; // FALLBACK: payload_OFF if models not loaded

    try {
        // Construct observation (mocked dimensions matching Python environment)
        // [soc, is_eclipse, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
        const s_t_raw = [
            satState.soc / 100, 
            satState.is_eclipse ? 1.0 : 0.0,
            satState.x / 7500e3, satState.y / 7500e3, satState.z / 7500e3,
            satState.vx / 8000, satState.vy / 8000, satState.vz / 8000
        ];
        
        const s_t_norm = normalizeState(s_t_raw);
        const s_t_tensor = new ort.Tensor('float32', new Float32Array(s_t_norm), [1, 8]);

        // World Model Step: Predict future (k=5)
        const s_future = new Float32Array(5 * 8); // Mocked for speed in JS UI
        const s_future_tensor = new ort.Tensor('float32', s_future, [1, 5, 8]);

        // Run Actor
        const results = await actor_session.run({ s_t: s_t_tensor, s_future: s_future_tensor });
        const logits = results.logits.data;
        
        // Argmax action
        let maxIdx = 0;
        for (let i = 1; i < logits.length; i++) {
            if (logits[i] > logits[maxIdx]) maxIdx = i;
        }
        
        return maxIdx; // Actions: 0=NOP, 1=OFF, 2=RELAY, 3=CHARGE, 4=HIBERNATE
    } catch (e) {
        return 1;
    }
}
