/*!
# cuda-instruction-set

Agent-native instruction set unifying FLUX VM, cuda-genepool instincts,
and cuda-axiom deliberation into one bytecode format.

## Opcode Categories
- Control (0x00-0x07): JMP, CALL, RET, HALT
- ArithConf (0x08-0x17): Arithmetic with confidence propagation
- Logic (0x18-0x1F): Bitwise with confidence
- Compare (0x20-0x27): Confidence-aware comparisons
- Stack (0x28-0x2F): Typed stack operations
- Perception (0x30-0x37): IO, sensor fusion
- A2A (0x38-0x4F): Agent-to-agent communication
- Memory (0x50-0x57): Capability-based regions
- Type (0x58-0x5F): Boxed values, casting
- SIMD (0x60-0x67): 4-wide vector operations
- Instinct (0x68-0x6F): Biological instinct activation
- Energy (0x70-0x77): ATP budgets, apoptosis
- System (0x78-0x7F): Debug, barriers, resources

## Confidence Propagation
Every ArithConf operation propagates uncertainty using Bayesian fusion:
  post_conf = 1 / (1/conf_a + 1/conf_b)
This ensures confidence decreases monotonically through computation chains.
*/

use serde::{Deserialize, Serialize};
use std::fmt;

/// Confidence value in [0.0, 1.0]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Confidence(pub f32);

impl Confidence {
    pub const ZERO: Confidence = Confidence(0.0);
    pub const CERTAIN: Confidence = Confidence(1.0);
    pub const HIGH: Confidence = Confidence(0.95);
    pub const MEDIUM: Confidence = Confidence(0.5);
    pub const LOW: Confidence = Confidence(0.1);

    pub fn new(v: f32) -> Self {
        Confidence(v.clamp(0.0, 1.0))
    }

    /// Bayesian fusion of two independent confidences
    pub fn fuse(a: Confidence, b: Confidence) -> Confidence {
        if a.0 <= 0.0 && b.0 <= 0.0 { return Confidence::ZERO; }
        if a.0 <= 0.0 { return b; }
        if b.0 <= 0.0 { return a; }
        Confidence::new(1.0 / (1.0 / a.0 + 1.0 / b.0))
    }

    /// Sequential composition: confidence decreases
    pub fn chain(a: Confidence, b: Confidence) -> Confidence {
        Confidence::new(a.0 * b.0)
    }
}

/// Opcode categories
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpCategory {
    Control, ArithConf, Logic, Compare, Stack,
    Perception, A2A, Memory, Type, SIMD,
    Instinct, Energy, System,
}

/// All agent opcodes (80 instructions)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Opcode {
    // Control 0x00-0x07
    Nop          = 0x00,
    Mov          = 0x01,
    MovI         = 0x02,
    Jmp          = 0x03,
    Jz           = 0x04,
    Jnz          = 0x05,
    Call         = 0x06,
    Ret          = 0x07,

    // ArithConf 0x08-0x17 (confidence-propagating)
    CAdd         = 0x08,
    CSub         = 0x09,
    CMul         = 0x0A,
    CDiv         = 0x0B,
    CMod         = 0x0C,
    CNeg         = 0x0D,
    CInc         = 0x0E,
    CDec         = 0x0F,
    CMin         = 0x10,
    CMax         = 0x11,
    CAbs         = 0x12,
    CLerp        = 0x13,
    ConfSet      = 0x14,
    ConfFuse     = 0x15,
    ConfChain    = 0x16,
    ConfThreshold = 0x17,

    // Logic 0x18-0x1F
    And          = 0x18,
    Or           = 0x19,
    Xor          = 0x1A,
    Not          = 0x1B,
    Shl          = 0x1C,
    Shr          = 0x1D,
    RotL         = 0x1E,
    RotR         = 0x1F,

    // Compare 0x20-0x27
    Cmp          = 0x20,
    Eq           = 0x21,
    Lt           = 0x22,
    Gt           = 0x23,
    Le           = 0x24,
    Ge           = 0x25,
    ConfEq       = 0x26,
    Test         = 0x27,

    // Stack 0x28-0x2F
    Push         = 0x28,
    Pop          = 0x29,
    Dup          = 0x2A,
    Swap         = 0x2B,
    Rot3         = 0x2C,
    Enter        = 0x2D,
    Leave        = 0x2E,
    Pick         = 0x2F,

    // Perception 0x30-0x37
    IoRead       = 0x30,
    IoWrite      = 0x31,
    SensorAcquire = 0x32,
    FuseConf     = 0x33,
    Perceive     = 0x34,
    Sense        = 0x35,
    Load8        = 0x36,
    Store8       = 0x37,

    // A2A 0x38-0x4F
    Tell         = 0x38,
    Ask          = 0x39,
    Delegate     = 0x3A,
    DelegResult  = 0x3B,
    Broadcast    = 0x3C,
    Reduce       = 0x3D,
    TrustCheck   = 0x3E,
    TrustUpdate  = 0x3F,
    TrustQuery   = 0x40,
    TrustRevoke  = 0x41,
    CapRequire   = 0x42,
    CapGrant     = 0x43,
    CapRevoke    = 0x44,
    DeclareIntent = 0x45,
    AssertGoal   = 0x46,
    VerifyOutcome = 0x47,
    Barrier      = 0x48,
    Formation    = 0x49,
    SyncClock    = 0x4A,
    ReportStatus = 0x4B,
    ExplainFail  = 0x4C,
    SetPriority  = 0x4D,
    RequestOverride = 0x4E,
    EmergencyStop = 0x4F,

    // Memory 0x50-0x57
    RegionCreate = 0x50,
    RegionDestroy = 0x51,
    RegionTransfer = 0x52,
    MemCopy      = 0x53,
    MemSet       = 0x54,
    MemCmp       = 0x55,
    Load         = 0x56,
    Store        = 0x57,

    // Type 0x58-0x5F
    Cast         = 0x58,
    Box          = 0x59,
    Unbox        = 0x5A,
    CheckType    = 0x5B,
    CheckBounds  = 0x5C,
    Tag          = 0x5D,
    Untag        = 0x5E,
    IsNil        = 0x5F,

    // SIMD 0x60-0x67 (4-wide i32 + confidence)
    VLoad        = 0x60,
    VStore       = 0x61,
    VAdd         = 0x62,
    VSub         = 0x63,
    VMul         = 0x64,
    VDiv         = 0x65,
    VFma         = 0x66,
    VConfFuse    = 0x67,

    // Instinct 0x68-0x6F (biological operations)
    InstinctActivate = 0x68,
    InstinctQuery = 0x69,
    GeneExpress  = 0x6A,
    EnzymeBind   = 0x6B,
    RnaTranslate = 0x6C,
    ProteinFold  = 0x6D,
    MembraneCheck = 0x6E,
    Quarantine   = 0x6F,

    // Energy 0x70-0x77
    AtpGenerate  = 0x70,
    AtpConsume   = 0x71,
    AtpQuery     = 0x72,
    AtpTransfer  = 0x73,
    ApoptosisCheck = 0x74,
    ApoptosisTrigger = 0x75,
    CircadianSet = 0x76,
    CircadianGet = 0x77,

    // System 0x78-0x7F
    Halt         = 0x78,
    Yield        = 0x79,
    ResourceAcquire = 0x7A,
    ResourceRelease = 0x7B,
    Debug        = 0x7C,
    DumpState    = 0x7D,
    Trap         = 0x7E,
    NopSys       = 0x7F,
}

impl Opcode {
    pub fn from_byte(b: u8) -> Option<Self> {
        // SAFETY: all values 0x00-0x7F are valid opcodes
        if b <= 0x7F { Some(unsafe { std::mem::transmute(b) }) } else { None }
    }

    pub fn to_byte(self) -> u8 { self as u8 }

    pub fn category(self) -> OpCategory {
        match self as u8 {
            0x00..=0x07 => OpCategory::Control,
            0x08..=0x17 => OpCategory::ArithConf,
            0x18..=0x1F => OpCategory::Logic,
            0x20..=0x27 => OpCategory::Compare,
            0x28..=0x2F => OpCategory::Stack,
            0x30..=0x37 => OpCategory::Perception,
            0x38..=0x4F => OpCategory::A2A,
            0x50..=0x57 => OpCategory::Memory,
            0x58..=0x5F => OpCategory::Type,
            0x60..=0x67 => OpCategory::SIMD,
            0x68..=0x6F => OpCategory::Instinct,
            0x70..=0x77 => OpCategory::Energy,
            0x78..=0x7F => OpCategory::System,
            _ => OpCategory::Control,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Opcode::Nop => "NOP", Opcode::Mov => "MOV", Opcode::MovI => "MOVI",
            Opcode::Jmp => "JMP", Opcode::Jz => "JZ", Opcode::Jnz => "JNZ",
            Opcode::Call => "CALL", Opcode::Ret => "RET",
            Opcode::CAdd => "CADD", Opcode::CSub => "CSUB", Opcode::CMul => "CMUL",
            Opcode::CDiv => "CDIV", Opcode::CMod => "CMOD", Opcode::CNeg => "CNEG",
            Opcode::CInc => "CINC", Opcode::CDec => "CDEC", Opcode::CMin => "CMIN",
            Opcode::CMax => "CMAX", Opcode::CAbs => "CABS", Opcode::CLerp => "CLERP",
            Opcode::ConfSet => "CONF_SET", Opcode::ConfFuse => "CONF_FUSE",
            Opcode::ConfChain => "CONF_CHAIN", Opcode::ConfThreshold => "CONF_THRESH",
            Opcode::And => "AND", Opcode::Or => "OR", Opcode::Xor => "XOR",
            Opcode::Not => "NOT", Opcode::Shl => "SHL", Opcode::Shr => "SHR",
            Opcode::RotL => "ROTL", Opcode::RotR => "ROTR",
            Opcode::Cmp => "CMP", Opcode::Eq => "EQ", Opcode::Lt => "LT",
            Opcode::Gt => "GT", Opcode::Le => "LE", Opcode::Ge => "GE",
            Opcode::ConfEq => "CONF_EQ", Opcode::Test => "TEST",
            Opcode::Push => "PUSH", Opcode::Pop => "POP", Opcode::Dup => "DUP",
            Opcode::Swap => "SWAP", Opcode::Rot3 => "ROT3", Opcode::Enter => "ENTER",
            Opcode::Leave => "LEAVE", Opcode::Pick => "PICK",
            Opcode::IoRead => "IO_READ", Opcode::IoWrite => "IO_WRITE",
            Opcode::SensorAcquire => "SENSOR_ACQ", Opcode::FuseConf => "FUSE_CONF",
            Opcode::Perceive => "PERCEIVE", Opcode::Sense => "SENSE",
            Opcode::Load8 => "LOAD8", Opcode::Store8 => "STORE8",
            Opcode::Tell => "TELL", Opcode::Ask => "ASK", Opcode::Delegate => "DELEGATE",
            Opcode::DelegResult => "DELEG_RESULT", Opcode::Broadcast => "BROADCAST",
            Opcode::Reduce => "REDUCE", Opcode::TrustCheck => "TRUST_CHECK",
            Opcode::TrustUpdate => "TRUST_UPDATE", Opcode::TrustQuery => "TRUST_QUERY",
            Opcode::TrustRevoke => "TRUST_REVOKE", Opcode::CapRequire => "CAP_REQ",
            Opcode::CapGrant => "CAP_GRANT", Opcode::CapRevoke => "CAP_REVOKE",
            Opcode::DeclareIntent => "DECLARE_INTENT", Opcode::AssertGoal => "ASSERT_GOAL",
            Opcode::VerifyOutcome => "VERIFY_OUTCOME", Opcode::Barrier => "BARRIER",
            Opcode::Formation => "FORMATION", Opcode::SyncClock => "SYNC_CLOCK",
            Opcode::ReportStatus => "REPORT_STATUS", Opcode::ExplainFail => "EXPLAIN_FAIL",
            Opcode::SetPriority => "SET_PRIORITY", Opcode::RequestOverride => "REQ_OVERRIDE",
            Opcode::EmergencyStop => "EMERGENCY_STOP",
            Opcode::RegionCreate => "REGION_CREATE", Opcode::RegionDestroy => "REGION_DESTROY",
            Opcode::RegionTransfer => "REGION_TRANSFER", Opcode::MemCopy => "MEMCOPY",
            Opcode::MemSet => "MEMSET", Opcode::MemCmp => "MEMCMP",
            Opcode::Load => "LOAD", Opcode::Store => "STORE",
            Opcode::Cast => "CAST", Opcode::Box => "BOX", Opcode::Unbox => "UNBOX",
            Opcode::CheckType => "CHECK_TYPE", Opcode::CheckBounds => "CHECK_BOUNDS",
            Opcode::Tag => "TAG", Opcode::Untag => "UNTAG", Opcode::IsNil => "IS_NIL",
            Opcode::VLoad => "VLOAD", Opcode::VStore => "VSTORE",
            Opcode::VAdd => "VADD", Opcode::VSub => "VSUB",
            Opcode::VMul => "VMUL", Opcode::VDiv => "VDIV",
            Opcode::VFma => "VFMA", Opcode::VConfFuse => "VCONF_FUSE",
            Opcode::InstinctActivate => "INSTINCT_ACT", Opcode::InstinctQuery => "INSTINCT_Q",
            Opcode::GeneExpress => "GENE_EXPR", Opcode::EnzymeBind => "ENZYME_BIND",
            Opcode::RnaTranslate => "RNA_TRANS", Opcode::ProteinFold => "PROTEIN_FOLD",
            Opcode::MembraneCheck => "MEMBRANE_CHK", Opcode::Quarantine => "QUARANTINE",
            Opcode::AtpGenerate => "ATP_GEN", Opcode::AtpConsume => "ATP_CONSUME",
            Opcode::AtpQuery => "ATP_Q", Opcode::AtpTransfer => "ATP_TRANSFER",
            Opcode::ApoptosisCheck => "APOPTOSIS_CHK", Opcode::ApoptosisTrigger => "APOPTOSIS_TRIGGER",
            Opcode::CircadianSet => "CIRCADIAN_SET", Opcode::CircadianGet => "CIRCADIAN_GET",
            Opcode::Halt => "HALT", Opcode::Yield => "YIELD",
            Opcode::ResourceAcquire => "RES_ACQ", Opcode::ResourceRelease => "RES_REL",
            Opcode::Debug => "DEBUG", Opcode::DumpState => "DUMP",
            Opcode::Trap => "TRAP", Opcode::NopSys => "NOP_SYS",
        }
    }
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", self.name()) }
}

/// A value with attached confidence
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ConfValue {
    pub value: i64,
    pub confidence: Confidence,
}

impl ConfValue {
    pub fn certain(v: i64) -> Self { ConfValue { value: v, confidence: Confidence::CERTAIN } }
    pub fn uncertain(v: i64, c: f32) -> Self { ConfValue { value: v, confidence: Confidence::new(c) } }
    pub fn zero() -> Self { ConfValue { value: 0, confidence: Confidence::CERTAIN } }
}

/// Variable-length instruction
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Instruction {
    pub opcode: Opcode,
    pub operands: Vec<u8>,
}

impl Instruction {
    pub fn new(op: Opcode, ops: Vec<u8>) -> Self { Instruction { opcode: op, operands: ops } }
    pub fn simple(op: Opcode) -> Self { Instruction { opcode: op, operands: vec![] } }
    pub fn reg(op: Opcode, rd: u8) -> Self { Instruction { opcode: op, operands: vec![rd] } }
    pub fn reg2(op: Opcode, rd: u8, rs: u8) -> Self { Instruction { opcode: op, operands: vec![rd, rs] } }
    pub fn reg_imm(op: Opcode, rd: u8, imm: i16) -> Self {
        Instruction { opcode: op, operands: vec![rd, (imm & 0xFF) as u8, ((imm >> 8) & 0xFF) as u8] }
    }

    /// Encode to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = vec![self.opcode.to_byte()];
        // Variable-length payload: 2-byte length prefix for complex ops
        match self.opcode.category() {
            OpCategory::A2A | OpCategory::Memory | OpCategory::Type => {
                let len = self.operands.len() as u16;
                bytes.push((len & 0xFF) as u8);
                bytes.push(((len >> 8) & 0xFF) as u8);
                bytes.extend(&self.operands);
            }
            _ => { bytes.extend(&self.operands); }
        }
        bytes
    }

    /// Decode from bytes, returns (instruction, bytes_consumed)
    pub fn decode(data: &[u8]) -> Option<(Self, usize)> {
        if data.is_empty() { return None; }
        let op = Opcode::from_byte(data[0])?;
        let (operands, consumed) = match op.category() {
            OpCategory::A2A | OpCategory::Memory | OpCategory::Type => {
                if data.len() < 3 { return None; }
                let len = (data[1] as u16) | ((data[2] as u16) << 8);
                let end = 3 + len as usize;
                if data.len() < end { return None; }
                (data[3..end].to_vec(), end)
            }
            OpCategory::Control | OpCategory::ArithConf | OpCategory::Logic
            | OpCategory::Compare | OpCategory::Stack | OpCategory::SIMD => {
                // Fixed format: opcode + operands (1-3 bytes typical)
                let op_len = match op {
                    Opcode::MovI | Opcode::Jmp | Opcode::Jz | Opcode::Jnz
                    | Opcode::Call => 4, // op + rd + imm16
                    Opcode::Nop | Opcode::Ret | Opcode::Halt | Opcode::Yield
                    | Opcode::Dup | Opcode::NopSys => 1,
                    Opcode::Enter | Opcode::Leave => 2, // op + frame_size
                    _ => 3, // op + rd + rs
                };
                let ops = if data.len() >= op_len { data[1..op_len].to_vec() } else { vec![] };
                (ops, op_len)
            }
            _ => (vec![], 1),
        };
        Some((Instruction { opcode: op, operands }, consumed))
    }
}

/// Simple text assembler
pub struct Assembler {
    pub instructions: Vec<Instruction>,
    pub labels: std::collections::HashMap<String, usize>,
    pub unresolved: Vec<(String, usize)>,
}

impl Assembler {
    pub fn new() -> Self { Assembler { instructions: vec![], labels: std::collections::HashMap::new(), unresolved: vec![] } }

    pub fn emit(&mut self, insn: Instruction) { self.instructions.push(insn); }

    pub fn label(&mut self, name: &str) { self.labels.insert(name.to_string(), self.instructions.len()); }

    pub fn emit_jmp(&mut self, label: &str) {
        self.unresolved.push((label.to_string(), self.instructions.len()));
        self.emit(Instruction::new(Opcode::Jmp, vec![0, 0, 0])); // placeholder
    }

    pub fn assemble(&mut self) -> Result<Vec<u8>, String> {
        // Resolve labels
        for (label, idx) in &self.unresolved {
            if let Some(&target) = self.labels.get(label) {
                let offset = target as i16 - *idx as i16;
                if let Some(ref mut insn) = self.instructions.get_mut(*idx) {
                    insn.operands = vec![0, (offset & 0xFF) as u8, ((offset >> 8) & 0xFF) as u8];
                }
            } else { return Err(format!("Unresolved label: {}", label)); }
        }
        // Encode all
        let mut bytecode = vec![];
        for insn in &self.instructions { bytecode.extend(insn.encode()); }
        Ok(bytecode)
    }
}

/// Text disassembler
pub struct Disassembler;

impl Disassembler {
    pub fn disassemble(data: &[u8]) -> Vec<String> {
        let mut lines = vec![];
        let mut pc = 0;
        while pc < data.len() {
            if let Some((insn, consumed)) = Instruction::decode(&data[pc..]) {
                let ops_str = insn.operands.iter().map(|b| format!("{}", b)).collect::<Vec<_>>().join(" ");
                if ops_str.is_empty() { lines.push(format!("{:04X}: {}", pc, insn.opcode.name())); }
                else { lines.push(format!("{:04X}: {} {}", pc, insn.opcode.name(), ops_str)); }
                pc += consumed;
            } else { lines.push(format!("{:04X}: ???", pc)); pc += 1; }
        }
        lines
    }
}

/// A2A message encoding
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct A2AMessage {
    pub sender_id: u32,
    pub intent: String,
    pub confidence: Confidence,
    pub payload_hash: u64,
    pub trust_level: f32,
    pub priority: u8,
}

impl A2AMessage {
    pub fn encode(&self) -> Vec<u8> {
        let intent_bytes = self.intent.as_bytes();
        let mut out = vec![];
        out.extend(&self.sender_id.to_le_bytes());
        out.push(intent_bytes.len() as u8);
        out.extend(intent_bytes);
        out.extend(&self.confidence.0.to_le_bytes());
        out.extend(&self.payload_hash.to_le_bytes());
        out.extend(&self.trust_level.to_le_bytes());
        out.push(self.priority);
        out
    }

    pub fn decode(data: &[u8]) -> Option<Self> {
        if data.len() < 4 + 1 + 4 + 8 + 4 + 1 { return None; }
        let sender_id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let intent_len = data[4] as usize;
        if data.len() < 5 + intent_len + 4 + 8 + 4 + 1 { return None; }
        let intent = String::from_utf8_lossy(&data[5..5 + intent_len]).to_string();
        let mut offset = 5 + intent_len;
        let confidence = f32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]);
        offset += 4;
        let payload_hash = u64::from_le_bytes(data[offset..offset+8].try_into().ok()?);
        offset += 8;
        let trust_level = f32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]);
        offset += 4;
        let priority = data[offset];
        Some(A2AMessage { sender_id, intent, confidence: Confidence::new(confidence), payload_hash, trust_level, priority })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opcode_roundtrip() {
        for b in 0..=0x7F {
            let op = Opcode::from_byte(b).unwrap();
            assert_eq!(op.to_byte(), b);
            assert_eq!(op.category(), Opcode::Nop.category() if b == 0 else op.category());
        }
        assert!(Opcode::from_byte(0x80).is_none());
    }

    #[test]
    fn test_confidence_fusion() {
        let a = Confidence::new(0.8);
        let b = Confidence::new(0.5);
        let fused = Confidence::fuse(a, b);
        // 1/(1/0.8 + 1/0.5) = 1/(1.25 + 2.0) = 1/3.25 ≈ 0.308
        assert!((fused.0 - 0.308).abs() < 0.01);
    }

    #[test]
    fn test_confidence_chain() {
        let a = Confidence::new(0.9);
        let b = Confidence::new(0.8);
        let chained = Confidence::chain(a, b);
        assert!((chained.0 - 0.72).abs() < 0.01);
    }

    #[test]
    fn test_instruction_encode_decode() {
        let insn = Instruction::reg2(Opcode::CAdd, 0, 1);
        let bytes = insn.encode();
        assert_eq!(bytes, vec![0x08, 0, 1]);
        let (decoded, consumed) = Instruction::decode(&bytes).unwrap();
        assert_eq!(decoded.opcode, Opcode::CAdd);
        assert_eq!(decoded.operands, vec![0, 1]);
        assert_eq!(consumed, 3);
    }

    #[test]
    fn test_movi_encode_decode() {
        let insn = Instruction::reg_imm(Opcode::MovI, 0, -42);
        let bytes = insn.encode();
        let (decoded, _) = Instruction::decode(&bytes).unwrap();
        assert_eq!(decoded.opcode, Opcode::MovI);
        assert_eq!(decoded.operands[0], 0);
        // imm -42 = 0xFFD6
        assert_eq!(decoded.operands[1], 0xD6);
        assert_eq!(decoded.operands[2], 0xFF);
    }

    #[test]
    fn test_a2a_message_roundtrip() {
        let msg = A2AMessage {
            sender_id: 42,
            intent: "investigate_anomaly".to_string(),
            confidence: Confidence::new(0.85),
            payload_hash: 12345,
            trust_level: 0.9,
            priority: 3,
        };
        let encoded = msg.encode();
        let decoded = A2AMessage::decode(&encoded).unwrap();
        assert_eq!(decoded.sender_id, 42);
        assert_eq!(decoded.intent, "investigate_anomaly");
        assert!((decoded.confidence.0 - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_assembler_basic() {
        let mut asm = Assembler::new();
        asm.emit(Instruction::reg_imm(Opcode::MovI, 0, 10));
        asm.emit(Instruction::reg_imm(Opcode::MovI, 1, 25));
        asm.emit(Instruction::reg2(Opcode::CAdd, 0, 1));
        asm.emit(Instruction::simple(Opcode::Halt));
        let bytecode = asm.assemble().unwrap();
        assert_eq!(bytecode[0], 0x02); // MOVI
        assert_eq!(bytecode[bytecode.len()-1], 0x78); // HALT
    }

    #[test]
    fn test_assembler_with_labels() {
        let mut asm = Assembler::new();
        asm.emit(Instruction::reg_imm(Opcode::MovI, 0, 1));
        asm.label("loop");
        asm.emit(Instruction::simple(Opcode::Nop));
        asm.emit(Instruction::new(Opcode::Jz, vec![0, 3, 0]));
        asm.emit_jmp("loop");
        let bytecode = asm.assemble().unwrap();
        assert!(!bytecode.is_empty());
    }

    #[test]
    fn test_disassembler() {
        let bytecode = vec![0x00, 0x78, 0x02, 0, 10, 0];
        let lines = Disassembler::disassemble(&bytecode);
        assert_eq!(lines[0], "0000: NOP");
        assert_eq!(lines[1], "0001: HALT");
        assert!(lines[2].starts_with("0002: MOVI"));
    }

    #[test]
    fn test_opcode_names() {
        assert_eq!(Opcode::Halt.name(), "HALT");
        assert_eq!(Opcode::CAdd.name(), "CADD");
        assert_eq!(Opcode::InstinctActivate.name(), "INSTINCT_ACT");
        assert_eq!(Opcode::AtpGenerate.name(), "ATP_GEN");
    }

    #[test]
    fn test_confidence_clamping() {
        assert_eq!(Confidence::new(1.5).0, 1.0);
        assert_eq!(Confidence::new(-0.5).0, 0.0);
    }

    #[test]
    fn test_all_categories_covered() {
        let cats = [
            OpCategory::Control, OpCategory::ArithConf, OpCategory::Logic,
            OpCategory::Compare, OpCategory::Stack, OpCategory::Perception,
            OpCategory::A2A, OpCategory::Memory, OpCategory::Type,
            OpCategory::SIMD, OpCategory::Instinct, OpCategory::Energy,
            OpCategory::System,
        ];
        for cat in &cats {
            let found = (0..=0x7Fu8).any(|b| Opcode::from_byte(b).map(|o| o.category() == *cat).unwrap_or(false));
            assert!(found, "Category {:?} has no opcodes", cat);
        }
    }

    #[test]
    fn test_instinct_opcodes() {
        assert_eq!(Opcode::InstinctActivate as u8, 0x68);
        assert_eq!(Opcode::GeneExpress as u8, 0x6A);
        assert_eq!(Opcode::EnzymeBind as u8, 0x6B);
        assert_eq!(Opcode::ApoptosisCheck as u8, 0x74);
        assert_eq!(Opcode::AtpGenerate as u8, 0x70);
    }
}
