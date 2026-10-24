"""Feature engineering for the Decode & Detect AML detector.

Every function takes the parsed tables of one split and returns one row per
transaction. Nothing here reads labels, so the same code runs on train and test.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

XRP_TOKEN = "0x1111222211112222111122221111222211112222"
MIN, HOUR, DAY = 60, 3600, 86400


# --------------------------------------------------------------------------
# 1. Parsing
# --------------------------------------------------------------------------
def _wire(w):
    """Decode an XRPL amount: XRP drops string or IOU dict -> (asset key, units)."""
    if isinstance(w, str):
        return "XRP", int(w) / 1e6
    return f"{w['issuer']}:{w['currency']}", float(w["value"])


def load_split(directory):
    """Read one split and return (payments, legs, rates).

    payments: one row per transaction, with sender, receiver, delivered asset,
              SendMax ceiling and every amount valued in reference USD.
    legs:     one row per executed transfer leg, valued in reference USD.
    """
    d = Path(directory)
    tx = pd.read_parquet(d / "transactions.parquet")
    tr = pd.read_parquet(d / "transfers.parquet")
    rates = pd.read_csv(d / "market_rates.csv")
    price = {(int(t), a): p / 1e6 for t, a, p in zip(rates.TIMESTAMP, rates.ASSET, rates.PRICE_MICRO_USD)}
    usd = lambda units, asset, hour: units * price.get((hour, asset), np.nan) if asset else np.nan

    tx["TIMESTAMP"] = tx.TIMESTAMP.astype(np.int64)
    tx["hour_ts"] = tx.TIMESTAMP // HOUR * HOUR
    extra = tx.EXTRA_TRANSACTION_FIELDS.map(json.loads)
    delivered = tx.DELIVERED_AMOUNT.map(lambda s: _wire(json.loads(s)))
    sendmax = extra.map(lambda e: _wire(e["SendMax"]) if "SendMax" in e else (None, np.nan))
    pay = pd.DataFrame({
        "TRANSACTION_HASH": tx.TRANSACTION_HASH,
        "t": tx.TIMESTAMP,
        "sender": tx.ACCOUNT,
        "receiver": extra.map(lambda e: e["Destination"]),
        "dlv_asset": delivered.str[0],
        "dlv_units": delivered.str[1],
        "sm_asset": sendmax.str[0],
        "sm_units": sendmax.str[1],
        "has_paths_field": extra.map(lambda e: "Paths" in e),
        "fee_drops": tx.SOURCE_FEE.astype(float),
        "sequence": tx.SEQUENCE.astype(float),
        "hour_ts": tx.hour_ts,
    })
    pay["dlv_usd"] = [usd(u, a, h) for u, a, h in zip(pay.dlv_units, pay.dlv_asset, pay.hour_ts)]
    pay["sm_usd"] = [usd(u, a, h) for u, a, h in zip(pay.sm_units, pay.sm_asset, pay.hour_ts)]

    tr["TIMESTAMP"] = tr.TIMESTAMP.astype(np.int64)
    tr["TRANSFER_INDEX"] = tr.TRANSFER_INDEX.astype(int)
    tr["asset"] = np.where(tr.SYMBOL == "XRP", "XRP", tr.TOKEN_ADDRESS + ":" + tr.SYMBOL)
    tr["hour_ts"] = tr.TIMESTAMP // HOUR * HOUR
    tr["usd"] = [usd(v, a, h) for v, a, h in zip(tr.VALUE, tr.asset, tr.hour_ts)]
    legs = tr.sort_values(["TRANSACTION_HASH", "TRANSFER_INDEX"]).reset_index(drop=True)
    return pay, legs, rates


# --------------------------------------------------------------------------
# 2. Cleaning
# --------------------------------------------------------------------------
def clean(pay, legs):
    """Drop exact duplicates, keep successful payments, attach leg summaries."""
    pay = pay.drop_duplicates("TRANSACTION_HASH").copy()
    legs = legs.drop_duplicates(["TRANSACTION_HASH", "TRANSFER_INDEX"])
    g = legs.groupby("TRANSACTION_HASH", sort=False)
    summary = pd.DataFrame({
        "n_legs": g.size(),
        "src_asset": g.asset.first(),
        "src_units": g.VALUE.first(),
        "src_usd": g.usd.first(),          # first executed leg = actual cost
        "last_asset": g.asset.last(),
        "n_issuers": g.TOKEN_ADDRESS.agg(lambda s: s[s != XRP_TOKEN].nunique()),
        "n_assets_path": g.asset.nunique(),
        "lps": g.apply(lambda d: tuple(d.TO_ADDRESS.iloc[:-1])),  # liquidity providers touched
        "route": g.apply(lambda d: ">".join(d.asset.str[-3:])),
    })
    pay = pay.join(summary, on="TRANSACTION_HASH")
    # Payments with no price (should not happen) fall back to the delivered USD value.
    pay["src_usd"] = pay.src_usd.fillna(pay.dlv_usd)
    pay = pay.sort_values(["t", "TRANSACTION_HASH"]).reset_index(drop=True)
    return pay, legs


# --------------------------------------------------------------------------
# 3. Transaction-intrinsic features
# --------------------------------------------------------------------------
def intrinsic_features(pay):
    f = pd.DataFrame(index=pay.index)
    f["log_usd"] = np.log1p(pay.dlv_usd)
    f["n_legs"] = pay.n_legs
    f["is_path_payment"] = (pay.n_legs > 1).astype(int)
    f["is_xrp_bridge"] = (pay.n_legs == 3).astype(int)
    f["delivers_xrp"] = (pay.dlv_asset == "XRP").astype(int)
    f["n_issuers_in_path"] = pay.n_issuers
    sym = lambda s: s.str.split(":").str[-1]
    # Issuer hop: same currency name in and out, different issuer (USD@A -> USD@B).
    f["issuer_hop_same_ccy"] = ((pay.n_legs > 1) & (sym(pay.src_asset) == sym(pay.dlv_asset))
                                & (pay.src_asset != pay.dlv_asset)).astype(int)
    # Meaningless hop: converts and comes back to the exact same asset.
    f["meaningless_roundtrip_asset"] = ((pay.n_legs > 1) & (pay.src_asset == pay.dlv_asset)).astype(int)
    # Value preservation across the legs, in reference USD.
    f["value_retention"] = pay.dlv_usd / pay.src_usd
    f["conversion_cost_bps"] = 1e4 * (1 - f.value_retention)
    # Slippage vs the SendMax ceiling.
    f["sendmax_over_cost"] = pay.sm_usd / pay.src_usd
    f["sendmax_over_delivered"] = pay.sm_usd / pay.dlv_usd
    f["sendmax_headroom_usd"] = pay.sm_usd - pay.src_usd
    f["hour_of_day"] = pd.to_datetime(pay.t, unit="s").dt.hour
    f["fee_drops"] = pay.fee_drops
    f["sequence"] = pay.sequence
    f["usd_cents_round"] = ((pay.dlv_usd * 100).round() % 100 == 0).astype(int)
    return f


# --------------------------------------------------------------------------
# 4. Account windows: fan-in/out, bursts, age, path & bridge ratios
# --------------------------------------------------------------------------
class AccountIndex:
    """Per-account sorted arrays of sends and receives for fast window queries."""

    def __init__(self, pay):
        self.pay = pay
        self.sends = {a: g.index.to_numpy() for a, g in pay.groupby("sender")}
        self.recvs = {a: g.index.to_numpy() for a, g in pay.groupby("receiver")}
        self.t = pay.t.to_numpy()

    def window(self, table, account, lo, hi):
        idx = table.get(account)
        if idx is None:
            return np.empty(0, int)
        tt = self.t[idx]
        return idx[np.searchsorted(tt, lo, "left"):np.searchsorted(tt, hi, "right")]


def account_window_features(pay, ix, windows=(10 * MIN, HOUR, 6 * HOUR, DAY)):
    t = pay.t.to_numpy()
    s_arr, r_arr = pay.sender.to_numpy(), pay.receiver.to_numpy()
    path = (pay.n_legs > 1).to_numpy()
    bridge = (pay.n_legs == 3).to_numpy()
    issuers = pay.src_asset.str.split(":").str[0].to_numpy()
    out = defaultdict(list)
    for i in range(len(pay)):
        s, r, ti = s_arr[i], r_arr[i], t[i]
        for w in windows:
            tag = f"{w // MIN}m"
            ss = ix.window(ix.sends, s, ti - w, ti + w)           # sender's sends around t
            rr = ix.window(ix.recvs, r, ti - w, ti + w)           # receiver's receipts around t
            s_in = ix.window(ix.recvs, s, ti - w, ti)             # sender's receipts before
            r_out = ix.window(ix.sends, r, ti, ti + w)            # receiver's sends after
            out[f"snd_out_cnt_{tag}"].append(len(ss))
            out[f"snd_out_uniq_{tag}"].append(len(set(r_arr[ss])))
            out[f"rcv_in_cnt_{tag}"].append(len(rr))
            out[f"rcv_in_uniq_{tag}"].append(len(set(s_arr[rr])))
            u_in, u_out = len(set(s_arr[s_in])), len(set(r_arr[ss]))
            out[f"snd_fanin_fanout_ratio_{tag}"].append((u_in + 1) / (u_out + 1))
            ui, uo = len(set(s_arr[rr])), len(set(r_arr[r_out]))
            out[f"rcv_fanin_fanout_ratio_{tag}"].append((ui + 1) / (uo + 1))
            out[f"rcv_fwd_cnt_{tag}"].append(len(r_out))
            out[f"snd_path_ratio_{tag}"].append(path[ss].mean() if len(ss) else 0)
            out[f"snd_bridge_ratio_{tag}"].append(bridge[ss].mean() if len(ss) else 0)
            out[f"snd_issuers_{tag}"].append(len(set(issuers[ss])))
    f = pd.DataFrame(out, index=pay.index)

    # Account age (first seen in the window) vs current activity.
    first_seen = pd.concat([pay[["sender", "t"]].rename(columns={"sender": "a"}),
                            pay[["receiver", "t"]].rename(columns={"receiver": "a"})]).groupby("a").t.min()
    week = pay.groupby("sender").size()
    span = max(t.max() - t.min(), 1)
    f["snd_age_h"] = (t - pay.sender.map(first_seen).to_numpy()) / HOUR
    f["rcv_age_h"] = (t - pay.receiver.map(first_seen).to_numpy()) / HOUR
    rate_1h = pay.sender.map(week).to_numpy() / span * HOUR
    f["snd_burst_vs_week"] = f["snd_out_cnt_60m"] / np.maximum(rate_1h, 1e-6)
    f["snd_week_sends"] = pay.sender.map(week).fillna(0).to_numpy()
    f["rcv_week_recvs"] = pay.receiver.map(pay.groupby("receiver").size()).fillna(0).to_numpy()
    f["snd_week_path_ratio"] = pay.sender.map(pay.assign(p=path).groupby("sender").p.mean()).to_numpy()
    f["snd_week_bridge_ratio"] = pay.sender.map(pay.assign(b=bridge).groupby("sender").b.mean()).to_numpy()
    lp = pay.assign(k=pay.lps.map(len)).groupby("sender").k.mean()
    f["snd_week_lp_per_tx"] = pay.sender.map(lp).to_numpy()

    # Relationship history: is this pair new? Recurring commerce re-uses pairs.
    pair = pay.sender + ">" + pay.receiver
    f["pair_week_cnt"] = pair.map(pair.value_counts()).to_numpy()
    f["pair_week_days"] = pair.map(pay.assign(d=t // DAY, p=pair).groupby("p").d.nunique()).to_numpy()
    f["pair_seq_no"] = pay.assign(p=pair).groupby("p").cumcount().to_numpy()
    rev = pay.receiver + ">" + pay.sender
    f["pair_reverse_exists"] = rev.isin(set(pair)).astype(int).to_numpy()
    return f


# --------------------------------------------------------------------------
# 5. Pass-through, chains, round trips
# --------------------------------------------------------------------------
def passthrough_features(pay, ix):
    t = pay.t.to_numpy()
    v = pay.dlv_usd.to_numpy()
    s_arr, r_arr = pay.sender.to_numpy(), pay.receiver.to_numpy()
    n = len(pay)
    out = defaultdict(list)
    for i in range(n):
        s, r, ti = s_arr[i], r_arr[i], t[i]
        nxt = ix.window(ix.sends, r, ti, ti + 3 * DAY)
        nxt = nxt[nxt != i]
        prv = ix.window(ix.recvs, s, ti - 3 * DAY, ti)
        prv = prv[prv != i]
        # Receiver: delay until it next pays someone, and how much of this money it moves on.
        out["rcv_next_send_delay"].append(t[nxt[0]] - ti if len(nxt) else 3 * DAY)
        out["rcv_next_send_ratio"].append(v[nxt[0]] / v[i] if len(nxt) else 0)
        # Best similar-value forward (within 20%) inside 6h.
        near = nxt[(t[nxt] - ti) <= 6 * HOUR]
        ok = near[np.abs(v[near] / v[i] - 1) <= 0.2] if len(near) else near
        out["rcv_similar_fwd_delay"].append(t[ok[0]] - ti if len(ok) else 6 * HOUR + 1)
        # Fraction of the received value that leaves the receiver within 1h / 6h.
        for w, tag in ((HOUR, "1h"), (6 * HOUR, "6h")):
            m = nxt[(t[nxt] - ti) <= w]
            out[f"rcv_passthrough_{tag}"].append(v[m].sum() / v[i] if len(m) else 0)
        # Sender: how long since it was paid, and how much.
        out["snd_prev_recv_delay"].append(ti - t[prv[-1]] if len(prv) else 3 * DAY)
        out["snd_prev_recv_ratio"].append(v[i] / v[prv[-1]] if len(prv) else 0)
        m = prv[(ti - t[prv]) <= HOUR]
        out["snd_inflow_1h_over_value"].append(v[m].sum() / v[i] if len(m) else 0)
    f = pd.DataFrame(out, index=pay.index)
    f["log_rcv_next_send_delay"] = np.log1p(f.rcv_next_send_delay)
    f["log_snd_prev_recv_delay"] = np.log1p(f.snd_prev_recv_delay)
    f["min_hop_delay"] = np.minimum(f.rcv_next_send_delay, f.snd_prev_recv_delay)
    return f


def link_graph(pay, ix, w_pass, w_sib=None, keep=None):
    """Directed links between payments that look like the same flow.

    pass-through link: i -> j if receiver(i) == sender(j) and 0 <= t_j - t_i <= w_pass
    sibling link (optional): same sender or same receiver within w_sib (fan-out / fan-in)
    keep (optional): boolean mask; only payments in the mask are linked. We pass
    "fresh pair" payments (the pair transacts once all week): planted motifs use new
    pairs ~95% of the time, background commerce only ~28%, so slow medium flows can be
    linked over hours without merging into the background.
    """
    t = pay.t.to_numpy()
    s_arr, r_arr = pay.sender.to_numpy(), pay.receiver.to_numpy()
    src, dst = [], []
    for i in range(len(pay)):
        j = ix.window(ix.sends, r_arr[i], t[i], t[i] + w_pass)
        j = j[j != i]
        src += [i] * len(j)
        dst += list(j)
        if w_sib:
            for table, acct in ((ix.sends, s_arr[i]), (ix.recvs, r_arr[i])):
                j = ix.window(table, acct, t[i], t[i] + w_sib)
                j = j[j > i]
                src += [i] * len(j)
                dst += list(j)
    src, dst = np.array(src, int), np.array(dst, int)
    if keep is not None:
        ok = keep[src] & keep[dst]
        src, dst = src[ok], dst[ok]
    return src, dst


def _components(n, src, dst):
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(src, dst):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def chain_features(pay, ix, w=10 * MIN, tag="10m"):
    """Longest pass-through chain through each payment, and round trips back to the sender."""
    n = len(pay)
    src, dst = link_graph(pay, ix, w)
    t = pay.t.to_numpy()
    succ = defaultdict(list)
    pred = defaultdict(list)
    for a, b in zip(src, dst):
        succ[a].append(b)
        pred[b].append(a)
    order = np.argsort(t, kind="stable")
    back = np.zeros(n, int)
    for i in order:  # longest chain ending at i
        back[i] = 1 + max((back[p] for p in pred[i]), default=0)
    fwd = np.zeros(n, int)
    for i in order[::-1]:
        fwd[i] = 1 + max((fwd[c] for c in succ[i]), default=0)
    s_arr, r_arr = pay.sender.to_numpy(), pay.receiver.to_numpy()
    cyc = np.zeros(n, int)
    cyc_len = np.zeros(n, int)
    for i in range(n):  # BFS forward up to 8 hops: does money come back to the sender?
        frontier, seen, depth = [i], {i}, 0
        while frontier and depth < 8 and not cyc[i]:
            depth += 1
            nxt = []
            for k in frontier:
                for c in succ[k]:
                    if c in seen:
                        continue
                    seen.add(c)
                    if r_arr[c] == s_arr[i]:
                        cyc[i], cyc_len[i] = 1, depth + 1
                        break
                    nxt.append(c)
            frontier = nxt
    f = pd.DataFrame(index=pay.index)
    f[f"chain_back_{tag}"] = back
    f[f"chain_fwd_{tag}"] = fwd
    f[f"chain_total_{tag}"] = back + fwd - 1
    f[f"round_trip_{tag}"] = cyc
    f[f"round_trip_len_{tag}"] = cyc_len
    return f


def component_features(pay, ix, w_pass, w_sib, tag, keep=None):
    """Group payments into flow components and describe each component's topology.

    Illicit flows mix topologies (fan-out then chain then fan-in...), so we count how
    many accounts in the component play each role and how mixed the roles are.
    """
    n = len(pay)
    src, dst = link_graph(pay, ix, w_pass, w_sib, keep)
    comp = _components(n, src, dst)
    d = pay[["t", "sender", "receiver", "dlv_usd", "n_legs", "src_usd"]].copy()
    d["comp"] = comp
    d["pair"] = d.sender + ">" + d.receiver
    pair_cnt = d.pair.map(d.pair.value_counts())
    d["fresh_pair"] = (pair_cnt == 1).astype(int)
    rows = {}
    for c, g in d.groupby("comp"):
        if len(g) == 1:
            continue
        outd = g.groupby("sender").receiver.nunique()
        ind = g.groupby("receiver").sender.nunique()
        acc = set(g.sender) | set(g.receiver)
        through = set(g.sender) & set(g.receiver)
        fan_out_nodes = int((outd >= 3).sum())
        fan_in_nodes = int((ind >= 3).sum())
        chain_nodes = len([a for a in through if outd.get(a, 0) == 1 and ind.get(a, 0) == 1])
        roles = np.array([fan_out_nodes, fan_in_nodes, chain_nodes, len(through) - chain_nodes], float)
        p = roles / roles.sum() if roles.sum() else roles
        entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        # Account-level cycle: any account that both sends and receives money back to an earlier sender.
        gaps = np.diff(np.sort(g.t.to_numpy()))
        tot_in = g[g.receiver.isin(through)].dlv_usd.sum()
        tot_out = g[g.sender.isin(through)].dlv_usd.sum()
        rows[c] = dict(n_tx=len(g), n_acc=len(acc), n_through=len(through), duration=g.t.max() - g.t.min(),
                       gap_med=np.median(gaps), gap_max=gaps.max(), usd_sum=g.dlv_usd.sum(),
                       usd_max=g.dlv_usd.max(), usd_cv=g.dlv_usd.std() / max(g.dlv_usd.mean(), 1e-9),
                       max_out=outd.max(), max_in=ind.max(), fan_out_nodes=fan_out_nodes,
                       fan_in_nodes=fan_in_nodes, chain_nodes=chain_nodes, topo_entropy=entropy,
                       n_roles=int((roles > 0).sum()), path_frac=(g.n_legs > 1).mean(),
                       bridge_frac=(g.n_legs == 3).mean(), fresh_pair_frac=g.fresh_pair.mean(),
                       through_retention=tot_out / tot_in if tot_in else 0,
                       value_retention=g.dlv_usd.sum() / g.src_usd.sum(),
                       closes_loop=int(bool(set(g.receiver) & set(g[g.t == g.t.min()].sender))))
    cf = pd.DataFrame.from_dict(rows, orient="index")
    f = pd.DataFrame({"comp": comp}, index=pay.index).join(cf, on="comp").drop(columns="comp")
    f["n_tx"] = f.n_tx.fillna(1)
    f["position_in_comp"] = d.groupby("comp").t.rank(pct=True).to_numpy()
    f.columns = [f"cmp{tag}_{c}" for c in f.columns]
    return f, comp


# --------------------------------------------------------------------------
# 6. Relay / weekly hub features (hard cases)
# --------------------------------------------------------------------------
def hub_features(pay, ix):
    """Week-long behaviour of the accounts on each side of a payment.

    Relays receive in waves 1-2 days apart and pass the money on hours later.
    We describe each account's 'dwell' time (receipt -> next send) over the week.
    """
    t = pay.t.to_numpy()
    v = pay.dlv_usd.to_numpy()
    dwell = {}
    for a, sidx in ix.sends.items():
        ridx = ix.recvs.get(a)
        if ridx is None:
            continue
        rt = t[ridx]
        k = np.searchsorted(rt, t[sidx], "right") - 1
        ok = k >= 0
        dwell[a] = t[sidx][ok] - rt[k[ok]]
    stats = {a: (np.median(x), np.min(x), np.percentile(x, 90), len(x)) for a, x in dwell.items() if len(x)}
    f = pd.DataFrame(index=pay.index)
    for side in ("sender", "receiver"):
        st = pay[side].map(stats)
        f[f"{side[:3]}_dwell_med"] = st.map(lambda x: x[0] if isinstance(x, tuple) else np.nan)
        f[f"{side[:3]}_dwell_min"] = st.map(lambda x: x[1] if isinstance(x, tuple) else np.nan)
        f[f"{side[:3]}_dwell_p90"] = st.map(lambda x: x[2] if isinstance(x, tuple) else np.nan)
        days_in = pay.groupby("receiver").t.agg(lambda x: len(set(x // DAY)))
        days_out = pay.groupby("sender").t.agg(lambda x: len(set(x // DAY)))
        f[f"{side[:3]}_days_in"] = pay[side].map(days_in).fillna(0)
        f[f"{side[:3]}_days_out"] = pay[side].map(days_out).fillna(0)
        f[f"{side[:3]}_week_in_usd"] = pay[side].map(pay.groupby("receiver").dlv_usd.sum()).fillna(0)
        f[f"{side[:3]}_week_out_usd"] = pay[side].map(pay.groupby("sender").dlv_usd.sum()).fillna(0)
        f[f"{side[:3]}_week_net_flow"] = (f[f"{side[:3]}_week_in_usd"] - f[f"{side[:3]}_week_out_usd"]) / (
            f[f"{side[:3]}_week_in_usd"] + f[f"{side[:3]}_week_out_usd"] + 1)
    # Waves: bursts of incoming payments separated by > 12h quiet periods.
    def waves(times):
        times = np.sort(times)
        return 1 + int((np.diff(times) > 12 * HOUR).sum()) if len(times) else 0
    in_waves = pay.groupby("receiver").t.agg(waves)
    out_waves = pay.groupby("sender").t.agg(waves)
    for side in ("sender", "receiver"):
        f[f"{side[:3]}_in_waves"] = pay[side].map(in_waves).fillna(0)
        f[f"{side[:3]}_out_waves"] = pay[side].map(out_waves).fillna(0)
    # This payment's own dwell: sender received money how long before paying it out,
    # relative to the sender's usual dwell (suspicious distributed relays forward faster).
    this_dwell = np.full(len(pay), np.nan)
    for a, sidx in ix.sends.items():
        ridx = ix.recvs.get(a)
        if ridx is None:
            continue
        k = np.searchsorted(t[ridx], t[sidx], "right") - 1
        ok = k >= 0
        this_dwell[sidx[ok]] = t[sidx[ok]] - t[ridx][k[ok]]
    f["snd_dwell_this"] = this_dwell
    f["snd_dwell_this_vs_med"] = np.log1p(this_dwell) - np.log1p(f.sen_dwell_med)
    # Pair recurrence across days for the receiver's outgoing side (relay -> same outputs every wave).
    pair = pay.sender + ">" + pay.receiver
    pd_days = pay.assign(p=pair, d=pay.t // DAY).groupby("p").d.nunique()
    f["snd_pairs_recurring_frac"] = pay.sender.map(
        pay.assign(r=pair.map(pd_days) > 1).groupby("sender").r.mean()).to_numpy()
    f["rcv_in_pairs_recurring_frac"] = pay.receiver.map(
        pay.assign(r=pair.map(pd_days) > 1).groupby("receiver").r.mean()).to_numpy()

    # This payment's value against the sender's usual payment size (uneven aggregate relays).
    g = pay.groupby("sender").dlv_usd
    f["value_vs_sender_median"] = v / pay.sender.map(g.median()).to_numpy()
    f["value_vs_sender_max"] = v / pay.sender.map(g.max()).to_numpy()
    f["value_z_in_sender"] = (v - pay.sender.map(g.mean()).to_numpy()) / (pay.sender.map(g.std()).fillna(1).to_numpy() + 1e-9)
    return f


# --------------------------------------------------------------------------
# 6b. Amount-matched flows (medium cases) and whole-flow hop statistics (easy cases)
# --------------------------------------------------------------------------
def matched_links(pay, ix, w_pass, w_sib, ratio=(0.6, 1.25)):
    """Sparse links between fresh-pair payments that carry the same money onwards.

    pass-through: each payment j gets at most ONE predecessor, the sender's receipt within
                  w_pass whose value is closest to j's (value ratio inside `ratio`).
    sibling:      the nearest earlier fresh payment from the same sender / to the same
                  receiver within w_sib (fan-out / fan-in legs).
    Keeping a single best predecessor stops slow medium flows from merging with the background.
    Returns (src, dst, kind) with kind 0 = pass-through, 1 = sibling.
    """
    t = pay.t.to_numpy()
    v = pay.dlv_usd.to_numpy()
    s_arr, r_arr = pay.sender.to_numpy(), pay.receiver.to_numpy()
    pair = pay.sender + ">" + pay.receiver
    fresh = (pair.map(pair.value_counts()) == 1).to_numpy()
    src, dst, kind = [], [], []
    for j in np.where(fresh)[0]:
        c = ix.window(ix.recvs, s_arr[j], t[j] - w_pass, t[j])
        c = c[(c != j) & fresh[c]]
        if len(c):
            rr = v[j] / v[c]
            c = c[(rr >= ratio[0]) & (rr <= ratio[1])]
            if len(c):
                src.append(c[np.argmin(np.abs(np.log(v[j] / v[c])))]); dst.append(j); kind.append(0)
        if w_sib:
            for table, acct in ((ix.sends, s_arr[j]), (ix.recvs, r_arr[j])):
                c = ix.window(table, acct, t[j] - w_sib, t[j])
                c = c[(c < j) & fresh[c]]
                if len(c):
                    src.append(c[-1]); dst.append(j); kind.append(1)
    return np.array(src, int), np.array(dst, int), np.array(kind, int)


def hop_stats(pay, comp, src, dst, tag):
    """Per-component statistics of the pass-through hops (src -> dst) inside it.

    Easy flows: nearly every hop is fast and keeps ~96% of the value.
    Decoys:     some hops are fast, others slow, and they lose more per hop.
    """
    t = pay.t.to_numpy()
    v = pay.dlv_usd.to_numpy()
    same = comp[src] == comp[dst]
    h = pd.DataFrame({"c": comp[src[same]], "delay": t[dst[same]] - t[src[same]],
                      "ret": v[dst[same]] / v[src[same]]})
    g = h.groupby("c")
    st = pd.DataFrame({
        "hops": g.size(),
        "hop_delay_med": g.delay.median(),
        "hop_delay_max": g.delay.max(),
        "hop_fast3m_share": g.delay.agg(lambda x: (x <= 180).mean()),
        "hop_ret_med": g.ret.median(),
        "hop_ret_min": g.ret.min(),
        "hop_ret_iqr": g.ret.quantile(0.75) - g.ret.quantile(0.25),
        "hop_ret_near_whole": g.ret.agg(lambda x: x.between(0.88, 1.08).mean()),
    })
    f = pd.DataFrame({"c": comp}, index=pay.index).join(st, on="c").drop(columns="c")
    f.columns = [f"{tag}_{c}" for c in f.columns]
    return f


def flow_features(pay, ix, w_pass, w_sib, tag, ratio=(0.6, 1.25)):
    """Amount-matched flow components with topology, timing, value and hop statistics."""
    src, dst, kind = matched_links(pay, ix, w_pass, w_sib, ratio)
    comp = _components(len(pay), src, dst)
    d = pay[["t", "sender", "receiver", "dlv_usd"]].assign(c=comp)
    rows = {}
    for c, g in d.groupby("c"):
        if len(g) == 1:
            continue
        through = set(g.sender) & set(g.receiver)
        outd = g.groupby("sender").receiver.nunique()
        ind = g.groupby("receiver").sender.nunique()
        gaps = np.diff(np.sort(g.t.to_numpy()))
        rows[c] = dict(n_tx=len(g), n_acc=len(set(g.sender) | set(g.receiver)), n_through=len(through),
                       duration=g.t.max() - g.t.min(), gap_med=np.median(gaps), gap_max=gaps.max(),
                       usd_sum=g.dlv_usd.sum(), usd_max=g.dlv_usd.max(),
                       usd_cv=g.dlv_usd.std() / max(g.dlv_usd.mean(), 1e-9),
                       max_out=outd.max(), max_in=ind.max(),
                       closes_loop=int(bool(set(g.receiver) & set(g.sender[g.t == g.t.min()]))))
    f = pd.DataFrame({"c": comp}, index=pay.index).join(pd.DataFrame.from_dict(rows, orient="index"), on="c")
    f["n_tx"] = f.n_tx.fillna(1)
    f["usd_per_hour"] = f.usd_sum / (f.duration / HOUR + 0.1)
    f["position"] = d.groupby("c").t.rank(pct=True).to_numpy()
    f = f.drop(columns="c")
    f.columns = [f"{tag}_{c}" for c in f.columns]
    pt = kind == 0
    return pd.concat([f, hop_stats(pay, comp, src[pt], dst[pt], tag)], axis=1), comp


def whole_flow_features(pay, ix, X):
    """Easy vs decoy: is the fast 5-minute burst the whole flow, or a fragment of a slower one?"""
    src, dst = link_graph(pay, ix, 5 * MIN)
    comp5 = _components(len(pay), *link_graph(pay, ix, 5 * MIN, 5 * MIN))
    f = hop_stats(pay, comp5, src, dst, "cmp5m")
    for tag in ("mflowA", "mflowB"):
        f[f"ctx_5m_over_{tag}_ntx"] = X.cmp5m_n_tx / X[f"{tag}_n_tx"]
        f[f"ctx_{tag}_slow_tail"] = X[f"{tag}_duration"].fillna(0) - X.cmp5m_duration.fillna(0)
    return f


# --------------------------------------------------------------------------
# 7. Everything together

# --------------------------------------------------------------------------
def build_features(pay, verbose=True):
    ix = AccountIndex(pay)
    parts = [intrinsic_features(pay)]
    log = print if verbose else (lambda *a, **k: None)
    log("intrinsic done"); parts.append(account_window_features(pay, ix))
    log("account windows done"); parts.append(passthrough_features(pay, ix))
    log("pass-through done")
    parts.append(chain_features(pay, ix, 10 * MIN, "10m"))
    parts.append(chain_features(pay, ix, HOUR, "1h"))
    log("chains done")
    comps = {}
    pair = pay.sender + ">" + pay.receiver
    fresh = (pair.map(pair.value_counts()) == 1).to_numpy()
    for w_pass, w_sib, tag, keep in ((5 * MIN, 5 * MIN, "5m", None), (30 * MIN, 10 * MIN, "30m", None),
                                     (HOUR, 30 * MIN, "F1h", fresh), (4 * HOUR, HOUR, "F4h", fresh)):
        cf, comps[tag] = component_features(pay, ix, w_pass, w_sib, tag, keep)
        parts.append(cf)
    log("components done"); parts.append(hub_features(pay, ix))
    log("hubs done")
    # Amount-matched flows: A = pure pass-through chains (4h hops), B = chains + fan siblings.
    fa, comps["mflowA"] = flow_features(pay, ix, 4 * HOUR, 0, "mflowA", ratio=(0.5, 1.5))
    fb, comps["mflowB"] = flow_features(pay, ix, 3 * HOUR, 20 * MIN, "mflowB")
    parts += [fa, fb]
    X = pd.concat(parts, axis=1)
    X = pd.concat([X, whole_flow_features(pay, ix, X)], axis=1)
    log("matched flows done")
    X = pd.concat([X, scenario_aggregates(pay, X, comps), normalised_graph_features(X),
                   hub_scenario_features(pay, ix), wide_round_trip_features(pay, comps)], axis=1)
    log("scenario aggregates done")
    # v9: a wider (fuller but noisier) medium grouping, then shape-relative scenario features.
    fc, comps["mflowC"] = flow_features(pay, ix, 4 * HOUR, HOUR, "mflowC")
    X = pd.concat([X, fc], axis=1)
    X = pd.concat([X, shape_relative_features(pay, comps)], axis=1)
    log("shape-relative done")
    X = apply_size_cap(X, comps)
    X.insert(0, "TRANSACTION_HASH", pay.TRANSACTION_HASH.to_numpy())
    return X, comps


# --------------------------------------------------------------------------
# 6d. Blob fix (v9): no "scenario" statistics for components too big to be a scenario
# --------------------------------------------------------------------------
SCENARIO_CAP = 40   # the largest planted scenario has 28 payments
CAP_PREFIXES = {
    "5m": ("cmp5m_", "scn5m_", "norm5m_"),
    "30m": ("cmp30m_", "norm30m_"),
    "F1h": ("cmpF1h_", "normF1h_"),
    "F4h": ("cmpF4h_", "scnF4h_", "normF4h_", "rt_F4h_"),
    "mflowA": ("mflowA_", "normmflowA_"),
    "mflowB": ("mflowB_", "scnmflowB_", "normmflowB_", "rt_mflowB_"),
    "mflowC": ("mflowC_",),
}


SHAPE_TAGS = ("5m", "F4h", "mflowB", "mflowC")


def shape_relative_features(pay, comps, cap=SCENARIO_CAP):
    """Medium cases (v9): compare each reconstructed scenario with look-alikes of the SAME shape.

    Medium laundering copies the shape of innocent decoys but is slower, moves more money and
    passes on more of it. Raw duration or totals vary a lot between shapes, so we rank them
    *within* scenarios of the same shape (payments, accounts, fan/chain roles) in the same split.
    The ranking uses no labels and is computed separately for train and test.

    Also describes each scenario's conversion mix (share of conversions, XRP bridges, distinct
    routes, route entropy, issuer hops), since medium cases use more varied conversions.
    """
    t = pay.t.to_numpy()
    v = pay.dlv_usd.to_numpy()
    s_arr, r_arr = pay.sender.to_numpy(), pay.receiver.to_numpy()
    sym = lambda s: s.str.split(":").str[-1]
    issuer_hop = ((pay.n_legs > 1) & (sym(pay.src_asset) == sym(pay.dlv_asset)) & (pay.src_asset != pay.dlv_asset)).to_numpy()
    out = []
    for tag in SHAPE_TAGS:
        comp = np.asarray(comps[tag])
        size = pd.Series(comp).map(pd.Series(comp).value_counts()).to_numpy()
        ok = (size >= 3) & (size <= cap)
        rows = {}
        for c in np.unique(comp[ok]):
            idx = np.where(comp == c)[0]
            idx = idx[np.argsort(t[idx], kind="stable")]
            snd, rcv = s_arr[idx], r_arr[idx]
            through = set(snd) & set(rcv)
            outd = pd.Series(rcv).groupby(snd).nunique()
            ind = pd.Series(snd).groupby(rcv).nunique()
            # per-hop "% passed on": each payment out of an intermediary vs the latest payment into it
            rets = []
            for k, i in enumerate(idx):
                prev = [j for j in idx[:k] if r_arr[j] == s_arr[i]]
                if prev:
                    rets.append(v[i] / v[prev[-1]])
            gaps = np.diff(t[idx])
            routes = pay.route.to_numpy()[idx]
            rc = pd.Series(routes).value_counts(normalize=True).to_numpy()
            n_acc = len(set(snd) | set(rcv))
            shape = (len(idx), n_acc, min(len(through), 4), int(outd.max() >= 3), int(ind.max() >= 3))
            rows[c] = dict(shape=str(shape), duration=t[idx].max() - t[idx].min(),
                           gap_med=np.median(gaps), usd_sum=v[idx].sum(), usd_max=v[idx].max(),
                           pass_on_med=np.median(rets) if rets else np.nan,
                           pass_on_near_whole=np.mean([0.88 <= r <= 1.08 for r in rets]) if rets else np.nan,
                           conv_frac=(pay.n_legs.to_numpy()[idx] > 1).mean(),
                           bridge_frac=(pay.n_legs.to_numpy()[idx] == 3).mean(),
                           n_routes=len(rc), route_entropy=float(-(rc * np.log(rc)).sum()),
                           issuer_hop_frac=issuer_hop[idx].mean(),
                           chainlike=int(outd.max() == 1 and ind.max() == 1))
        sc = pd.DataFrame.from_dict(rows, orient="index")
        if len(sc):
            shape_n = sc.groupby("shape").shape.transform("size")
            ntx = sc["shape"].str.extract(r"\((\d+)")[0].astype(int)
            for col in ("duration", "gap_med", "usd_sum", "usd_max", "pass_on_med", "conv_frac", "route_entropy"):
                by_shape = sc.groupby("shape")[col].rank(pct=True)
                by_size = sc.groupby(ntx)[col].rank(pct=True)
                sc[f"{col}_pct_shape"] = np.where(shape_n >= 5, by_shape, by_size)   # fall back to same size
            sc["shape_count"] = shape_n
            # medium signature: slower AND richer than look-alikes; pass-on only meaningful for chain-like flows
            sc["slow_rich_score"] = sc.duration_pct_shape + sc.usd_sum_pct_shape
            sc["chainlike_pass_on"] = np.where(sc.chainlike == 1, sc.pass_on_med, np.nan)
            sc = sc.drop(columns="shape")
        f = pd.DataFrame({"c": comp}, index=pay.index).join(sc, on="c").drop(columns="c")
        f.columns = [f"shp{tag}_{c}" for c in f.columns]
        out.append(f)
    return pd.concat(out, axis=1)


def apply_size_cap(X, comps, cap=SCENARIO_CAP):
    """Blank out component features where the component is larger than any real scenario.

    Long link windows chain unrelated payments into one giant component (~4,600 payments in
    train, ~3,500 in test). Its blob-wide statistics are identical for thousands of payments,
    differ completely between train and test, and span every CV fold, so the model memorised
    them and CV over-stated v8. Missing values tell LightGBM "no reliable scenario here".
    """
    X = X.copy()
    big = {}
    for tag, comp in comps.items():
        c = pd.Series(comp, index=X.index)
        big[tag] = (c.map(c.value_counts()) > cap).to_numpy()
    for tag, prefixes in CAP_PREFIXES.items():
        if tag not in big:
            continue
        cols = [c for c in X.columns if c.startswith(prefixes)]
        X.loc[big[tag], cols] = np.nan
    for c in [c for c in X.columns if c.startswith("ctx_")]:
        tags = [t for t in big if f"_{t}_" in c + "_"]
        if tags:
            X.loc[np.any([big[t] for t in tags], axis=0), c] = np.nan
    return X


# --------------------------------------------------------------------------
# 6c. Scenario-level view (v8): aggregate every payment's features over its whole flow
# --------------------------------------------------------------------------
SCENARIO_AGG_COLS = ["log_min_hop", "rcv_passthrough_6h", "snd_inflow_1h_over_value", "pair_week_cnt",
                     "log_usd", "n_legs", "rcv_next_send_ratio", "log_snd_dwell_this", "hour_of_day"]


def scenario_aggregates(pay, X, comps):
    """Treat each flow component as a reconstructed scenario and describe it as a whole.

    For every payment we attach statistics of ALL payments in its scenario: totals, how regular
    the payment timing is (coefficient of variation of the gaps), and the mean / spread of the
    per-payment features. All payments of a scenario share one label, so the model should see
    the scenario, not only the single payment.
    """
    base = pd.DataFrame({
        "log_min_hop": np.log1p(X.min_hop_delay), "rcv_passthrough_6h": X.rcv_passthrough_6h,
        "snd_inflow_1h_over_value": X.snd_inflow_1h_over_value, "pair_week_cnt": X.pair_week_cnt,
        "log_usd": X.log_usd, "n_legs": X.n_legs, "rcv_next_send_ratio": X.rcv_next_send_ratio.clip(0, 5),
        "log_snd_dwell_this": np.log1p(X.snd_dwell_this), "hour_of_day": X.hour_of_day,
    }, index=X.index)
    t = pay.t.to_numpy()
    out = []
    for tag in ("5m", "F4h", "mflowB"):
        comp = pd.Series(comps[tag], index=X.index)
        size = comp.map(comp.value_counts())
        g = base.groupby(comp)
        mean = g.transform("mean").add_prefix(f"scn{tag}_mean_")
        std = g.transform("std").add_prefix(f"scn{tag}_std_")
        f = pd.concat([mean, std], axis=1)
        # payment-frequency regularity inside the scenario
        order = np.argsort(t, kind="stable")
        gaps = pd.Series(np.nan, index=X.index)
        prev = pd.Series(t[order], index=order).groupby(comp.iloc[order].to_numpy()).diff()
        gaps.loc[prev.index] = prev.to_numpy()
        gg = gaps.groupby(comp)
        f[f"scn{tag}_gap_cv"] = (gg.transform("std") / (gg.transform("mean") + 1)).to_numpy()
        f[f"scn{tag}_value_vs_scenario_mean"] = (pay.dlv_usd / pay.dlv_usd.groupby(comp).transform("mean")).to_numpy()
        f.loc[size.to_numpy() == 1, :] = np.nan   # singletons have no scenario
        out.append(f)
    return pd.concat(out, axis=1)


def normalised_graph_features(X):
    """Rework of low-importance topology counts: divide by the size of the graph they come from.

    A raw count (e.g. 3 fan-out nodes) means different things in a 5-payment flow and a
    500-payment component, so we express roles, density and pace relative to graph size.
    """
    f = pd.DataFrame(index=X.index)
    for tag in ("5m", "30m", "F1h", "F4h"):
        n_acc = X[f"cmp{tag}_n_acc"]
        n_tx = X[f"cmp{tag}_n_tx"]
        for role in ("fan_out_nodes", "fan_in_nodes", "chain_nodes", "n_through"):
            f[f"norm{tag}_{role}_per_acc"] = X[f"cmp{tag}_{role}"] / n_acc
        f[f"norm{tag}_tx_per_acc"] = n_tx / n_acc
        f[f"norm{tag}_sec_per_tx"] = X[f"cmp{tag}_duration"] / n_tx
        f[f"norm{tag}_usd_per_tx"] = X[f"cmp{tag}_usd_sum"] / n_tx
        f[f"norm{tag}_log_size"] = np.log1p(n_tx)
    for tag in ("mflowA", "mflowB"):
        f[f"norm{tag}_sec_per_tx"] = X[f"{tag}_duration"] / X[f"{tag}_n_tx"]
        f[f"norm{tag}_through_per_acc"] = X[f"{tag}_n_through"] / X[f"{tag}_n_acc"]
        f[f"norm{tag}_usd_per_tx"] = X[f"{tag}_usd_sum"] / X[f"{tag}_n_tx"]
    return f


def hub_scenario_features(pay, ix):
    """Relay view without rebuilding cohorts.

    For each account we collect its *recurring* links (pairs seen on >= 2 different days), the
    way relay scenarios repeat sources and outputs across waves. This "hub scenario" is then
    described as a whole (total, largest and unevenness of the amounts, hold time before
    forwarding), and ranked against hub scenarios of the same size, a stand-in for the
    relay cohorts, which share a fixed number of payments.
    """
    t = pay.t.to_numpy()
    pair = pay.sender + ">" + pay.receiver
    days = pair.map(pay.assign(p=pair, d=t // DAY).groupby("p").d.nunique())
    rec = (days >= 2).to_numpy()
    d = pay.loc[rec, ["t", "sender", "receiver", "dlv_usd"]]
    rows = {}
    accounts = set(d.sender) | set(d.receiver)
    ins = d.groupby("receiver")
    outs = d.groupby("sender")
    for a in accounts:
        i = ins.get_group(a) if a in ins.groups else d.iloc[:0]
        o = outs.get_group(a) if a in outs.groups else d.iloc[:0]
        if len(i) == 0 or len(o) == 0:
            continue
        it = np.sort(i.t.to_numpy())
        k = np.searchsorted(it, o.t.to_numpy(), "right") - 1
        hold = o.t.to_numpy()[k >= 0] - it[k[k >= 0]]
        allv = np.concatenate([i.dlv_usd.to_numpy(), o.dlv_usd.to_numpy()])
        rows[a] = dict(n=len(i) + len(o), n_src=i.sender.nunique(), n_dst=o.receiver.nunique(),
                       usd_in=i.dlv_usd.sum(), usd_max=allv.max(), usd_cv=allv.std() / allv.mean(),
                       hold_med=np.median(hold) if len(hold) else np.nan,
                       hold_min=hold.min() if len(hold) else np.nan,
                       waves=1 + int((np.diff(it) > 12 * HOUR).sum()))
    hs = pd.DataFrame.from_dict(rows, orient="index")
    if len(hs):
        # "cohort" proxy: hub scenarios with the same number of recurring payments
        for c in ("usd_in", "usd_max", "usd_cv", "hold_med", "hold_min"):
            hs[f"{c}_pct_in_size"] = hs.groupby("n")[c].rank(pct=True)
    f = pd.DataFrame(index=pay.index)
    for side in ("sender", "receiver"):
        j = pay[[side]].join(hs, on=side).drop(columns=side)
        j.columns = [f"hubscn_{side[:3]}_{c}" for c in j.columns]
        f = pd.concat([f, j], axis=1)
    f["hubscn_on_recurring_pair"] = rec.astype(int)
    return f


def wide_round_trip_features(pay, comps):
    """Rework of the dead round-trip family: look over the whole reconstructed flow, not 10 min.

    Does the sender of this payment receive money back later inside the same scenario, and how
    much of what it sent comes back?
    """
    f = pd.DataFrame(index=pay.index)
    for tag in ("F4h", "mflowB"):
        comp = comps[tag]
        d = pay[["t", "sender", "receiver", "dlv_usd"]].assign(c=comp)
        back_usd, back_any = np.zeros(len(d)), np.zeros(len(d))
        for c, g in d.groupby("c"):
            if len(g) < 3:
                continue
            for i, s, ti, v in zip(g.index, g.sender, g.t, g.dlv_usd):
                later = g[(g.receiver == s) & (g.t > ti)]
                if len(later):
                    back_any[i] = 1
                    back_usd[i] = later.dlv_usd.sum() / v
        f[f"rt_{tag}_returns"] = back_any
        f[f"rt_{tag}_return_ratio"] = back_usd
    return f


# --------------------------------------------------------------------------
# 8. Base rules as explicit 0/1 flags (evaluated on their own and fed to the model)
# --------------------------------------------------------------------------
BASE_RULES = {
    # name: (target cases, description)
    "R01_fast_connected_burst": ("easy", ">=4 linked payments inside a 5-min-linked component lasting <= 10 min"),
    "R02_burst_under_3min": ("easy", ">=3 linked payments, whole component inside 3 min (only ~10% of decoys are this fast)"),
    "R03_straight_passthrough": ("easy", "money passes straight through: intermediaries forward 85-115% of what they got"),
    "R04_fresh_pair_flow": ("easy/medium", ">=90% of the component's sender->receiver pairs are brand new this week"),
    "R05_slow_fresh_flow": ("medium", "4-30 payments linked over new pairs within 4h hops, lasting > 1h"),
    "R06_slow_flow_high_retention": ("medium", "R05 and intermediaries keep >= 90% of value (decoys ~83-88%)"),
    "R07_slow_flow_bigger_total": ("medium", "R05 and component moves more than the median R05 total"),
    "R08_round_trip": ("easy", "money comes back to the original sender within 1h of linked hops"),
    "R09_fast_relay_hub": ("hard (distributed relay)", "sender is a multi-wave hub and forwards within 1-3.2h of receipt"),
    "R10_relay_largest_payment": ("hard (aggregate relay)", "multi-wave hub sender, and this is its largest payment of the week"),
    "R11_issuer_hop": ("any", "converts between the same currency from two different issuers"),
    "R12_meaningless_roundtrip_asset": ("any", "converts and delivers exactly the asset it started with"),
    "R13_sendmax_slack": ("none expected", "SendMax ceiling above 1.01x the actual cost"),
    "R14_new_account_burst": ("easy", "sender first seen < 1h ago and already sent >= 3 payments in the hour"),
}


def rule_flags(X):
    r = pd.DataFrame(index=X.index)
    r["R01_fast_connected_burst"] = (X.cmp5m_n_tx >= 4) & (X.cmp5m_duration <= 600)
    r["R02_burst_under_3min"] = (X.cmp5m_n_tx >= 3) & (X.cmp5m_duration <= 180)
    r["R03_straight_passthrough"] = (X.cmp5m_n_through >= 1) & X.cmp5m_through_retention.between(0.85, 1.15)
    r["R04_fresh_pair_flow"] = (X.cmp30m_n_tx >= 4) & (X.cmp30m_fresh_pair_frac >= 0.9)
    slow = (X.cmpF4h_n_tx.between(4, 30)) & (X.cmpF4h_duration > HOUR)
    r["R05_slow_fresh_flow"] = slow
    r["R06_slow_flow_high_retention"] = slow & (X.cmpF4h_through_retention >= 0.9)
    r["R07_slow_flow_bigger_total"] = slow & (X.cmpF4h_usd_sum > X.cmpF4h_usd_sum[slow].median())
    r["R08_round_trip"] = X.round_trip_1h == 1
    hub = X.sen_in_waves >= 3
    r["R09_fast_relay_hub"] = hub & X.snd_dwell_this.between(HOUR, 3.2 * HOUR)
    r["R10_relay_largest_payment"] = hub & (X.value_vs_sender_max >= 0.999)
    r["R11_issuer_hop"] = X.issuer_hop_same_ccy == 1
    r["R12_meaningless_roundtrip_asset"] = X.meaningless_roundtrip_asset == 1
    r["R13_sendmax_slack"] = X.sendmax_over_cost > 1.0101
    r["R14_new_account_burst"] = (X.snd_age_h < 1) & (X.snd_out_cnt_60m >= 3)
    return r.astype(int)


# Which of the requested feature families each column belongs to (for importance plots).
def feature_group(col):
    rules = [
        ("Shape-relative scenarios", ("shp",)),
        ("Relay hub scenario", ("hubscn",)),
        ("Scenario aggregates", ("scn",)),
        ("Graph-size normalised", ("norm",)),
        ("Round-trip / circular", ("rt_",)),
        ("Base rules", ("R0", "R1")),
        ("Direct vs path ratios", ("snd_path_ratio", "snd_week_path_ratio", "is_path_payment", "n_legs", "cmp5m_path", "cmp30m_path", "cmpF1h_path", "cmpF4h_path")),
        ("Slippage / SendMax", ("sendmax_",)),
        ("Round-trip / circular", ("round_trip", "closes_loop", "pair_reverse")),
        ("Issuer hopping", ("n_issuers", "snd_issuers", "issuer_hop")),
        ("Meaningless hops", ("meaningless",)),
        ("Fan-in / fan-out velocity", ("snd_out_", "rcv_in_", "fanin_fanout", "max_out", "max_in", "fan_out_nodes", "fan_in_nodes")),
        ("Account age & bursts", ("_age_h", "burst", "week_sends", "week_recvs")),
        ("Bridge dependency", ("bridge", "lp_per_tx", "is_xrp_bridge", "delivers_xrp")),
        ("Value preservation", ("value_retention", "conversion_cost", "through_retention", "passthrough", "_ratio")),
        ("Pass-through timing", ("delay", "chain_", "dwell", "rcv_fwd")),
        ("Topology mix (components)", ("cmp",)),
        ("Relay / weekly hub", ("waves", "days_", "week_", "pair_", "recurring", "value_vs", "value_z")),
    ]
    for name, keys in rules:
        if any(k in col for k in keys):
            return name
    return "Transaction basics"
