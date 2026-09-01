import glob
import os


def select_files(folder, select_documents, count, exact=False):
    all_of = sorted(glob.glob(os.path.join(folder, "*")))
    print(f"Folder «{folder}» contains {len(all_of)} file(s) in total.")

    if select_documents:
        choice = [str(v).strip() for v in select_documents if str(v).strip()]
        if exact:
            choice_set = {os.path.splitext(v)[0] for v in choice}
            files = [f for f in all_of if os.path.splitext(os.path.basename(f))[0] in choice_set]
            missing = choice_set - {os.path.splitext(os.path.basename(f))[0] for f in all_of}
        else:
            files = [f for f in all_of if any(v in os.path.basename(f) for v in choice)]
            missing = [v for v in choice if not any(v in os.path.basename(f) for f in all_of)]
        print(f"Selected: {len(files)} file(s) matched {len(choice)} queries ({'exact' if exact else 'substring'})")
        if len(files) <= 5:
            for f in files:
                print("   ", os.path.basename(f))
        if missing:
            missing_names = sorted(missing) if isinstance(missing, set) else missing
            print(f"!! No match for {len(missing_names)} of the queries:", missing_names[:5])
    elif count in (None, 0) or str(count).strip().lower() in ("all", "every"):
        files = all_of
        print(f"Mode: ALL, running all {len(files)} files.")
    else:
        n = int(count)
        files = all_of[:n]
        print(f"Mode: COUNT, running the first {len(files)} of {len(all_of)} (asked for {n}).")

    return files
