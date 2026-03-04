
def csv_align_data(input_alignment, output_file, n_chains):
    alignment = SeqIO.parse(input_alignment, "fasta")
    df = pd.DataFrame(columns=["id", "sequence"])
    for rec in alignment:
        label_parts = rec.id.split("|")[1].split(".")
        red_label = f"{label_parts[0]}_{label_parts[1]}"
        seq_print = str(rec.seq)
        if n_chains > 1:
            # ColabFold reads multimer chains separated by ":"
            seq_print = ":".join([seq_print] * n_chains)
        dfi = pd.DataFrame.from_dict({"id": [red_label], "sequence": [seq_print]})
        df = pd.concat([df, dfi], ignore_index=True)
    df.to_csv(output_file, index=False)
    return output_file