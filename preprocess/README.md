# Preprocessing Script

The script is a quick preprocess of sketch image before calling model in AI platform.
The script is deployed on Cloud Function for easiness. It takes JSON `{"img": <base64 string of image>}` as input,
and return the request of JSON formated according to the model served on AI platform.
The preporcess steps are:
1. Crop the sketch to minimize white padding
1. Resize the sketch to 256 x 256.
1. Skeletonize the lines.
1. Encode the image to PNG format.
1. Create JSON string

## Deploy:
```sh
gcloud functions deploy preprocessing --entry-point preprocess --runtime python37 --trigger-http
```

## Input and output:
- Input: 
A JSON of `{"img": <base64 string of image>}`.The PNG sketch image should be encoded as BASE64 String,
and white background is required for the image:
```json
    {
    "img":"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CHWAAAECsaQkHtm85ZZbmpxir732ioA+cuTIs88++3Of+5yw3uTjpXkECPS7gKTe70OgAQQIEOitQGyRHivUJ06c2Nsb9eX3I6BPmzattO7l1ltvjXXqX/3qVzds2NCXdbo3AQIE8i3gidJ8j5/WEyBAIATOOOOMeJlR7H3e5BodHns97LDDDjrooEceeaTJm615BAgQ6C8Bc+r9Ja9eAgQINExgxYoVRx99dMNul+pG//zP/7xs2bJXX301VYXqIUCAQM4EJPWcDZjmEiBAoLNAvPtz6tSpnc8325n169fH+0rLrYrV6vGS1C984QvlMw4IECBAoFJAUq/UcEyAAIH8Cdx///1Dhgw566yzmr/p8Z7U9vb2ynZee+21S5YsmT9/fuVJxwQIECBQErBO3W8CAQIE8i0wadKkwYMHR97NRTf22Weff/iHf/jOd75Tbu2FF154zz33PP744x/5yEfKJx0QIECAQAhI6n4NCBAgkGOBhx56aObMmU888USTb/xSJv7MZz7zwgsvvPzyy+UzcfBXf/VXsYAnHoqtPOmYAAECBKx+8TtAgACBHAvEu4S2b9+el5ge0DGDHqvVO4g/9dRTb7zxxhe/+MUO530kQIBAiwtI6i3+C6D7BAjkWyD2Ttl3331z1IePfvSjXe6hHvvA/PznP+/yUo56p6kECBBorICk3lhPdyNAgEBSgVhJcuCBByatsneV7dixo62trXMi/8pXvhL7wMyYMaN3t/dtAgQIFEpAUi/UcOoMAQKtJrB69erDDz88R70eM2bMBz/4wS4TuX1gcjSOmkqAQBoBST2Ns1oIECDQJwKxvPuoo47qk1v32U2/+c1vPvvss52n1c8888xPfepT3/72t/usZjcmQIBAzgQk9ZwNmOYSIECgUmDz5s3HH3985ZnmP548efLOnTuHDx/euanXX3/9H/7wh9GjR3e+5AwBAgRaUEBSb8FB12UCBAoi8MADD8Sa7wi++erPiBEjYsuaLts8cuTIxx57LK4uWLCgywJOEiBAoKUE7KfeUsOtswQIFEogX+88KtOvXLlyypQpnVe/lAvMmjVr0aJFb775ZvmMAwIECLSmgKTemuOu1wQI5F7g6aefPu64455//vkcbaZeQl+3bt2oUaN27dpVbQx++9vffvjDH37ttde6XCFT7VvOEyBAoHgCknrxxlSPCBBo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"
    }
```
![input example](input.png)


- Trigger:
```sh
curl -X POST "https://<PROJECT_ID>.cloudfunctions.net/preprocessing " -H "Content-Type: application/json" --data '{"img": <base64 string of image>}'
```

- Output:
A JSON of `{'image_bytes': {"b64": <base64 string of image>}}`. Example of output:
```json
   {"image_bytes": {"b64": "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"}}
```
![output example](testb64.png)