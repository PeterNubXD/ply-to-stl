import gradio as gr
import pymeshlab
import tempfile
import os

POISSON_DEPTH = 9   # 8=fast/good, 9=high quality, 10=very high/slow


def convert_ply_to_stl(ply_file):
    if ply_file is None:
        raise gr.Error("Please upload a PLY file first.")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(ply_file)

    # Step 1: Remove isolated pieces (wrt Diameter)
    try:
        ms.apply_filter("meshing_remove_connected_component_by_diameter")
    except Exception:
        pass

    # Step 2: Clean duplicates and null faces
    try:
        ms.apply_filter("meshing_remove_duplicate_faces")
        ms.apply_filter("meshing_remove_duplicate_vertices")
        ms.apply_filter("meshing_remove_null_faces")
    except Exception:
        pass

    # Step 3: Compute normals (tries multiple names across pymeshlab versions)
    _normal_filter_names = [
        "compute_normal_for_point_clouds",
        "compute_normal_for_point_sets",
        "compute_normals_for_point_sets",
    ]
    _applied = False
    for _name in _normal_filter_names:
        try:
            ms.apply_filter(_name, k=10)
            _applied = True
            break
        except Exception:
            continue
    if not _applied:
        raise gr.Error("Could not compute normals. Please try again or contact support.")

    # Step 4: Screened Poisson reconstruction
    try:
        ms.apply_filter(
            "generate_surface_reconstruction_screened_poisson",
            depth=POISSON_DEPTH,
            preclean=True,
        )
    except Exception as e:
        raise gr.Error(f"Screened Poisson failed: {e}")

    # Step 5: Smooth
    for _ in range(7):
        ms.apply_filter("apply_coord_taubin_smoothing")
    for _ in range(6):
        ms.apply_filter("apply_coord_laplacian_smoothing")

    # Step 6: Export
    output_path = tempfile.mktemp(suffix=".stl")
    ms.save_current_mesh(output_path, binary=True)

    filename = os.path.basename(ply_file).replace(".ply", "_converted.stl")
    friendly_path = os.path.join(os.path.dirname(output_path), filename)
    os.rename(output_path, friendly_path)
    return friendly_path


with gr.Blocks(title="PLY → STL Converter") as app:
    gr.Markdown(
        """
        # PLY → STL Converter
        Upload a `.ply` file and download a high-quality `.stl`.
        Uses Screened Poisson reconstruction for a clean, solid surface.
        """
    )

    with gr.Row():
        with gr.Column():
            ply_input = gr.File(label="Upload PLY file", file_types=[".ply"])
            convert_btn = gr.Button("Convert", variant="primary")
        with gr.Column():
            stl_output = gr.File(label="Download STL")
            status = gr.Textbox(label="Status", interactive=False)

    def run_conversion(ply_file):
        if ply_file is None:
            return None, "No file uploaded."
        try:
            result = convert_ply_to_stl(ply_file)
            return result, "Done! Click the file above to download."
        except gr.Error as e:
            return None, f"Error: {e}"
        except Exception as e:
            return None, f"Unexpected error: {e}"

    convert_btn.click(
        fn=run_conversion,
        inputs=[ply_input],
        outputs=[stl_output, status],
    )

app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
