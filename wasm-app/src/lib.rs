use spin_sdk::http::{IntoResponse, Request, Response, Method};
use spin_sdk::http_component;
use tract_onnx::prelude::*;
use image::io::Reader as ImageReader;
use image::imageops::FilterType;
use image::GenericImageView;
use std::io::Cursor;

#[http_component]
fn handle_request(req: Request) -> anyhow::Result<impl IntoResponse> {
    match *req.method() {
        Method::Post => predict(req),
        _ => Ok(Response::builder().status(405).body("Method Not Allowed").build()),
    }
}

fn predict(req: Request) -> anyhow::Result<Response> {
    let body = req.body();
    let img_reader = ImageReader::new(Cursor::new(body)).with_guessed_format()?;
    let img = img_reader.decode()?;

    // Resize
    let resized = img.resize_exact(224, 224, FilterType::CatmullRom);

    // Normalize
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    
    let mut input_data = Vec::with_capacity(1 * 3 * 224 * 224);

    for pixel in resized.pixels() {
        let val = pixel.2[0] as f32 / 255.0;
        input_data.push((val - mean[0]) / std[0]);
    }
    for pixel in resized.pixels() {
        let val = pixel.2[1] as f32 / 255.0;
        input_data.push((val - mean[1]) / std[1]);
    }
    for pixel in resized.pixels() {
        let val = pixel.2[2] as f32 / 255.0;
        input_data.push((val - mean[2]) / std[2]);
    }

    // Load Model
    let model = tract_onnx::onnx()
        .model_for_path("mobilenetv2.onnx")?
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 224, 224)))?
        .into_optimized()?
        .into_runnable()?;

    // Run Inference
    let image_tensor = tract_ndarray::Array4::from_shape_vec((1, 3, 224, 224), input_data)?;
    let result = model.run(tvec!(image_tensor.into_tensor().into()))?;

    // Process Output
    let output_view = result[0].to_array_view::<f32>()?;
    let scores = output_view.as_slice().unwrap();

    // Find the single best score
    let mut best_class = 0;
    let mut best_score = -1.0;

    for (i, &score) in scores.iter().enumerate() {
        if score > best_score {
            best_score = score;
            best_class = i;
        }
    }

    // Return simple JSON
    let json = serde_json::json!({
        "class": best_class,
        "score": best_score,
        "runtime": "Spin + Tract (Pure Rust)"
    });

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(json.to_string())
        .build())
}