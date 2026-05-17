export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  kb_used?: boolean;
  duration_sec?: number;
}

export interface CognitiveProfile {
  user_id: string;
  n_turns_scored: number;
  dimension_profile: Record<string, {
    dominant_score: number | "NA";
    max_capability: number | "NA";
    n_responses: number;
    score_distribution: Record<string, number>;
  }>;
  icap_level_distribution: Record<string, number>;
  dominant_icap: string;
  max_icap_achieved: string;
  engagement_trajectory: string;
  avg_duration_sec: number;
  duration_causal_correlation: number | null;
}

export interface ReliabilityData {
  reliability: Record<string, {
    fleiss_kappa: number | null;
    krippendorff_alpha: number | null;
    pct_consensus: number;
    n_items: number;
    interpretation: string;
  }>;
  n_turns: number;
}

export type IcapLevel = "Passive" | "Active" | "Constructive" | "Interactive";

export interface UploadedFile {
  file_id: string;
  filename: string;
  file_type: "image" | "table" | "pdf" | "text";
  description_preview: string;
  /** local object URL for image preview, created by URL.createObjectURL */
  localPreviewUrl?: string;
  uploadProgress?: number; // 0-100 while uploading, undefined when done
  error?: string;
}