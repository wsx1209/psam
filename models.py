

class FeatureEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int, num_tokens: int, 
                 modality_id: int, num_types: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model, bias=True)
        self.roi_embed = nn.Embedding(num_tokens, d_model)
        self.mod_embed = nn.Embedding(2, d_model)
        self.type_embed = nn.Embedding(num_types, d_model)
        self.modality_id = modality_id
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor ,feature_types: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.proj(x)
        roi_idx = torch.arange(T, device=x.device)
        roi_emb = self.roi_embed(roi_idx)[None, :, :].expand(B, -1, -1)
        mod_ids = torch.full((B, T), self.modality_id, device=x.device, dtype=torch.long)
        mod_emb = self.mod_embed(mod_ids)
        type_emb = self.type_embed(feature_types)
        out = h + roi_emb + mod_emb + type_emb
        out = self.dropout(out)
        return out


class BiCrossModalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln_mri = nn.LayerNorm(d_model)
        self.ln_pet = nn.LayerNorm(d_model)
        self.attn_mri = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                               dropout=dropout, batch_first=True)
        self.attn_pet = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                               dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat_mri: torch.Tensor, feat_pet: torch.Tensor):
        q_mri = self.ln_mri(feat_mri)
        q_pet = self.ln_pet(feat_pet)
        mri_enh, _ = self.attn_mri(q_mri, q_pet, q_pet, need_weights=False)
        mri_enh = self.dropout(mri_enh) + feat_mri
        pet_enh, _ = self.attn_pet(q_pet, q_mri, q_mri, need_weights=False)
        pet_enh = self.dropout(pet_enh) + feat_pet
        return mri_enh, pet_enh


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.ln_mri = nn.LayerNorm(d_model)
        self.ln_pet = nn.LayerNorm(d_model)
        self.ln_mri_enh = nn.LayerNorm(d_model)
        self.ln_pet_enh = nn.LayerNorm(d_model)
        self.ln_cat = nn.LayerNorm(4 * d_model)
        self.proj = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ln_cls = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, feat_mri, feat_pet, feat_mri_enh, feat_pet_enh):
        mri_pool = self.ln_mri(feat_mri).mean(dim=1)
        pet_pool = self.ln_pet(feat_pet).mean(dim=1)
        mri_enh_pool = self.ln_mri_enh(feat_mri_enh).mean(dim=1)
        pet_enh_pool = self.ln_pet_enh(feat_pet_enh).mean(dim=1)
        x = torch.cat([mri_pool, pet_pool, mri_enh_pool, pet_enh_pool], dim=-1)
        x = self.ln_cat(x)
        x = self.proj(x)
        x = x.mean(dim=1)
        x = self.ln_cls(x)
        logits = self.classifier(x)
        return logits


class PDFBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, 
                 ffn_hidden_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.ln_in = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.ssm = Mamba2(d_model=d_model)
        self.gate_linear = nn.Linear(d_model * 3, 1)
        self.dropout = nn.Dropout(dropout)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden_factor * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_factor * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        residual = x
        h = self.ln_in(x)
        attn_out, _ = self.mhsa(h, h, h, attn_mask=attn_mask)
        ssm_out = self.ssm(h)
        delta_h = self.ln_in(attn_out - ssm_out)
        gate = torch.sigmoid(self.gate_linear(delta_h))
        fused = gate * attn_out + (1.0 - gate) * ssm_out
        fused = self.dropout(fused)
        y = residual + fused
        y_norm = self.ln_ff(y)
        ffn_out = self.ffn(y_norm)
        out = y + ffn_out
        out = self.ln_ff(out)
        return out


class PSAMModel(nn.Module):
    def __init__(self, in_dim_mri: int, in_dim_pet: int, num_tokens_mri: int,
                 num_tokens_pet: int, num_classes: int, d_model: int = 128,
                 num_pdf_layers: int = 4, n_heads: int = 4, dropout: float = 0.1 ,num_types: int = 2):
        super().__init__()
        self.encoder_mri = FeatureEncoder(in_dim_mri, d_model, num_tokens_mri, 0, num_types, dropout)
        self.encoder_pet = FeatureEncoder(in_dim_pet, d_model, num_tokens_pet, 1, num_types, dropout)
        self.mri_blocks = nn.ModuleList([
            PDFBlock(d_model, n_heads, 4, dropout) for _ in range(num_pdf_layers)
        ])
        self.pet_blocks = nn.ModuleList([
            PDFBlock(d_model, n_heads, 4, dropout) for _ in range(num_pdf_layers)
        ])
        self.bi_cma = BiCrossModalAttention(d_model, n_heads, dropout)
        self.head = ClassificationHead(d_model, num_classes, dropout)

    def forward(self, x_mri: torch.Tensor, x_pet: torch.Tensor, feature_types: torch.Tensor) -> torch.Tensor:
        h_mri = self.encoder_mri(x_mri, feature_types)
        h_pet = self.encoder_pet(x_pet, feature_types)
        for blk in self.mri_blocks:
            h_mri = blk(h_mri)
        for blk in self.pet_blocks:
            h_pet = blk(h_pet)
        h_mri_enh, h_pet_enh = self.bi_cma(h_mri, h_pet)
        logits = self.head(h_mri, h_pet, h_mri_enh, h_pet_enh)
        return logits
