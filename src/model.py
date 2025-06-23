from nesymres.architectures.model import Model
from nesymres.dclasses import FitParams, BFGSParams
from nesymres.architectures.data import de_tokenize
from nesymres.dataset.generator import Generator
from functools import partial
import torch
import torch.nn.functional as F
import sympy
import json
import omegaconf
from typing import Callable, Tuple, Dict
from dataclasses import dataclass

device = "cpu" # NOTE: change to cuda if your GPU can handle it
@dataclass
class ModelParams:
    model: Model
    fitfunc: Callable
    params_fit: FitParams
    bfgs: BFGSParams
    eq_cfg: Dict
    model_cfg: omegaconf.DictConfig

def load_model(
    eq_cfg_path="../res/100m_eq_cfg.json",
    model_cfg_path="../res/100m_cfg.yaml",
    checkpoint_path="../res/100m.ckpt",
    device="cuda" if torch.cuda.is_available() else "cpu",
) -> ModelParams:
    # load equation config
    with open(eq_cfg_path, "r") as json_file:
        eq_cfg = json.load(json_file)

    # load model config
    model_cfg = omegaconf.OmegaConf.load(model_cfg_path)

    # load other parameters
    bfgs = BFGSParams(
        activated= model_cfg.inference.bfgs.activated,
        n_restarts=model_cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=model_cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=model_cfg.inference.bfgs.normalization_o,
        idx_remove=model_cfg.inference.bfgs.idx_remove,
        normalization_type=model_cfg.inference.bfgs.normalization_type,
        stop_time=model_cfg.inference.bfgs.stop_time,
    )

    params_fit = FitParams(word2id=eq_cfg["word2id"], 
        id2word={int(k): v for k,v in eq_cfg["id2word"].items()}, 
        una_ops=eq_cfg["una_ops"], 
        bin_ops=eq_cfg["bin_ops"], 
        total_variables=list(eq_cfg["total_variables"]),  
        total_coefficients=list(eq_cfg["total_coefficients"]),
        rewrite_functions=list(eq_cfg["rewrite_functions"]),
        bfgs=bfgs,
        beam_size=model_cfg.inference.beam_size #This parameter is a tradeoff between accuracy and fitting time
    )

    # load model
    model = Model.load_from_checkpoint(checkpoint_path, cfg=model_cfg.architecture).to(device)
    model.eval()

    fitfunc = partial(model.fitfunc, cfg_params=params_fit)

    return ModelParams(model, fitfunc, params_fit, bfgs, eq_cfg, model_cfg)

# TODO: quit if all equations have reached a final token
@torch.no_grad()
def greedy_predict(
    model: Model, params: FitParams,
    X: torch.Tensor = None, y: torch.Tensor = None, sequence: torch.Tensor = None, enc_embed: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Greedily predicts the next token in the sequence. Can be used in two ways:
    1. If `X` and `y` are provided, the model will use them to compute the encoder embedding and then predict the next token.
    2. If `enc_embed` is provided, the model will use this embedding directly to predict the next token.

    # Args
    * `X`: Function domain of shape [B, N, D], where B is the batch size, N is the number of samples,
        and D the input dimensionality (no more than 3).
    * `y`: Function image corresponding to `X` of shape [B, N].
    * `sequence`: The initial tokens to predict the next token from of shape [B, S], where S is the maximum sequence length.
        All samples in the batch are expected to be at the same current sequence length.
    * `enc_embed`: Encoder embedding. Can be reused if the same `X` and `y` are used repeatedly for prediction.

    # Returns
    A tuple containing:
    * `next_token`: The predicted token IDs for each sample in the batch.
    * `sequence`: The sequence of tokens generated so far, which is updated with the predicted token.
    * `enc_embed`: The encoder embedding, which may be resued.
    """

    if enc_embed is None:
        # compute encoder embedding
        enc_input = torch.cat((X, y.unsqueeze(-1)), dim=-1).to(model.device)
        enc_embed = model.enc(enc_input)

    batch_size = enc_embed.size(0)

    if sequence is None:
        # initialize sequence with start token
        sequence = torch.zeros((batch_size, model.cfg.length_eq), dtype=torch.long, device=model.device)
        sequence[:, 0] = params.word2id["S"]

    cur_len = (sequence != 0).sum(dim=1).max().item()

    # generate decoder masks
    mask1, mask2 = model.make_trg_mask(
        sequence[:, :cur_len]
    )

    # compute positional embeddings
    pos = model.pos_embedding(
        torch.arange(0, cur_len)
            .unsqueeze(0)
            .repeat(sequence.shape[0], 1)
            .type_as(sequence)
    )

    # embed tokens
    seq_embed = model.tok_embedding(sequence[:, :cur_len])
    seq_embed = model.dropout(seq_embed + pos)

    # run decoder
    output = model.decoder_transfomer(
        seq_embed.permute(1, 0, 2),
        enc_embed.permute(1, 0, 2),
        mask2.float(),
        tgt_key_padding_mask=mask1.bool(),
    )
    output = model.fc_out(output)
    output = output.permute(1, 0, 2).contiguous()

    # add next token
    # NOTE: softmax not really necessary here, but may come in handy later
    token_probs = F.softmax(output[:, -1:, :], dim=-1).squeeze(1)
    next_token = torch.argmax(token_probs, dim=-1)
    sequence[:, cur_len] = next_token

    return next_token, sequence, enc_embed

def tokens_to_text(tokens: torch.Tensor, params: FitParams) -> list[str]:
    """
    Converts a batches of token IDs to their corresponding text representations.

    # Args
    * `tokens`: Of shape [B, S], where B is the batch size and S is the maximum sequence length.
    """

    equations = []
    for seq in tokens:
        raw = de_tokenize(seq[1:].tolist(), params.id2word)

        pretty_eq = sympy.sympify(
            Generator.prefix_to_infix(
                raw, 
                coefficients=["constant"], 
                variables=params.total_variables
            ).replace("{constant}", f"c",1)
        )
        equations.append(str(pretty_eq))

    return equations
