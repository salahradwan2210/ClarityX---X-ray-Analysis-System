"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { LucideFileText, LucideLoader } from "lucide-react"
import { useAuth } from "@/lib/auth-context"
import { useToast } from "@/hooks/use-toast"
import { jsPDF } from "jspdf"
import 'jspdf-autotable'
import autoTable from 'jspdf-autotable'
import { createHeatmap } from "@/lib/heatmap-utils"

interface PdfReportGeneratorProps {
  result: any
  patientData: any
}

export default function PdfReportGenerator({ result, patientData }: PdfReportGeneratorProps) {
  const [isGenerating, setIsGenerating] = useState(false)
  const { user, profile } = useAuth()
  const { toast } = useToast()

  // مواقع طبية دقيقة للأمراض المختلفة
  const medicalLocations = {
    "Atelectasis": {x: 0.35, y: 0.45, width: 0.3, height: 0.35},   // انخماص في الفصوص السفلية غالباً
    "Cardiomegaly": {x: 0.5, y: 0.5, width: 0.4, height: 0.3},     // تضخم القلب في وسط الصدر
    "Effusion": {x: 0.8, y: 0.65, width: 0.25, height: 0.25},      // انصباب في قاعدة الرئة اليمنى
    "Infiltration": {x: 0.4, y: 0.4, width: 0.4, height: 0.4},     // ارتشاح منتشر في الرئة
    "Mass": {x: 0.35, y: 0.35, width: 0.25, height: 0.25},         // كتلة غالباً في الفصوص العليا
    "Nodule": {x: 0.3, y: 0.3, width: 0.15, height: 0.15},         // عقيدة أصغر من الكتلة
    "Pneumonia": {x: 0.45, y: 0.45, width: 0.35, height: 0.35},    // التهاب رئوي عادةً في الفصوص السفلية
    "Pneumothorax": {x: 0.25, y: 0.4, width: 0.2, height: 0.4},    // استرواح صدري على حواف الرئة
    "Consolidation": {x: 0.45, y: 0.55, width: 0.3, height: 0.25}, // تكاثف في الفصوص السفلية
    "Edema": {x: 0.5, y: 0.5, width: 0.4, height: 0.4},            // وذمة منتشرة في الرئة
    "Emphysema": {x: 0.4, y: 0.3, width: 0.35, height: 0.3},       // نفاخ رئوي في الفصوص العليا
    "Fibrosis": {x: 0.45, y: 0.35, width: 0.3, height: 0.25},      // تليف في الفصوص الوسطى والعليا
    "Pleural_Thickening": {x: 0.7, y: 0.5, width: 0.2, height: 0.4}, // سماكة الغشاء البلوري على الحواف
    "Hernia": {x: 0.5, y: 0.7, width: 0.25, height: 0.2}           // فتق أسفل الحجاب الحاجز
  };

  const generatePdf = async () => {
    setIsGenerating(true)

    try {
      console.log("Starting PDF generation...");
      console.log("Patient data:", patientData);
      console.log("View position value:", patientData.viewPosition, patientData.view_position);
      
      // Create a new PDF document - A4 in portrait
      const doc = new jsPDF({
        orientation: "portrait",
        unit: "mm",
        format: "a4"
      });
      
      const pageWidth = doc.internal.pageSize.width;
      const pageHeight = doc.internal.pageSize.height;
      const margin = 20;
      let yPosition = margin;
      
      // Add header with logo and hospital info
      doc.setFillColor(41, 98, 255);
      doc.rect(0, 0, pageWidth, 30, "F");
      
      doc.setTextColor(255, 255, 255);
      doc.setFontSize(22);
      doc.setFont("helvetica", "bold");
      doc.text("DiagnoLink", margin, 15);
      
      doc.setFontSize(12);
      doc.setFont("helvetica", "normal");
      doc.text("Medical Imaging Analysis Report", margin, 22);

      // Add hospital info on the right
      const hospitalName = profile?.hospital || "Central Medical Center";
      doc.setFontSize(10);
      doc.text(hospitalName, pageWidth - margin - doc.getTextWidth(hospitalName), 15);
      
      const date = new Date().toLocaleDateString();
      doc.text(`Report Date: ${date}`, pageWidth - margin - doc.getTextWidth(`Report Date: ${date}`), 22);
      
      // Reset text color for the rest of the document
      doc.setTextColor(60, 60, 60);
      yPosition = 40;
      
      // Add report ID and reference
      doc.setFontSize(9);
      doc.setTextColor(100, 100, 100);
      doc.text(`Report ID: ${result.id.substring(0, 8)}`, margin, yPosition);
      doc.text(`Ref: XR-${new Date().getFullYear()}-${Math.floor(Math.random() * 10000)}`, pageWidth - margin - 40, yPosition);
      
      yPosition += 10;
      
      // Add section divider
      doc.setDrawColor(220, 220, 220);
      doc.setLineWidth(0.5);
      doc.line(margin, yPosition, pageWidth - margin, yPosition);
      
      yPosition += 10;
      
      // Patient Information Section
      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(41, 98, 255);
      doc.text("Patient Information", margin, yPosition);
      
      yPosition += 6;
      
      // Create a table for patient info
      doc.setFontSize(10);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(60, 60, 60);
      
      const patientInfoData = [
        ["Patient Name:", patientData.name || "Not available"],
        ["Patient ID:", patientData.id ? patientData.id.substring(0, 8) : "Not available"],
        ["Age:", `${patientData.age || "N/A"} years`],
        ["Gender:", patientData.gender === "male" ? "Male" : "Female"],
        ["View Position:", patientData.viewPosition || patientData.view_position || "Not available"],
        ["Examination Date:", new Date(result.created_at || Date.now()).toLocaleDateString()]
      ];
      
      autoTable(doc, {
        startY: yPosition,
        body: patientInfoData,
        theme: 'plain',
        styles: {
          fontSize: 10,
          cellPadding: 1,
        },
        columnStyles: {
          0: { fontStyle: 'bold', cellWidth: 30 },
          1: { cellWidth: 60 }
        },
        margin: { left: margin }
      });
      
      yPosition = (doc as any).lastAutoTable.finalY + 10;
      
      // Physician Information - Using actual logged-in user information
      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(41, 98, 255);
      doc.text("Physician Information", margin, yPosition);
      
      yPosition += 6;
      
      // Get doctor name from the logged-in user
      const doctorName = user?.email 
        ? (profile?.full_name || user.email.split('@')[0]) 
        : "Dr. Medical Professional";
      
      const physicianInfoData = [
        ["Physician Name:", doctorName],
        ["Specialty:", profile?.specialty || "Radiology"],
        ["License No.:", profile?.license_number || "MED-12345"],
        ["Contact:", profile?.phone_number || (user?.email || "Not available")]
      ];
      
      autoTable(doc, {
        startY: yPosition,
        body: physicianInfoData,
        theme: 'plain',
        styles: {
          fontSize: 10,
          cellPadding: 1,
        },
        columnStyles: {
          0: { fontStyle: 'bold', cellWidth: 30 },
          1: { cellWidth: 60 }
        },
        margin: { left: margin }
      });
      
      yPosition = (doc as any).lastAutoTable.finalY + 10;
      
      // Add X-ray images (original and with heatmap)
      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(41, 98, 255);
      doc.text("X-ray Images", margin, yPosition);
      
      // Add a separator line under the X-ray Images title
      doc.setDrawColor(220, 220, 220);
      doc.setLineWidth(0.3);
      doc.line(margin, yPosition + 4, pageWidth - margin, yPosition + 4);
      
      yPosition += 10; // Space after title
      
      // Add the original image to the first page
      const originalImg = new Image();
      originalImg.crossOrigin = "Anonymous";
      
      // Wait for the original image to load
      await new Promise((resolve, reject) => {
        originalImg.onload = resolve;
        originalImg.onerror = reject;
        originalImg.src = result.imageUrl;
      });
      
      // Calculate dimensions to fit within half the page width
      const maxWidth = (pageWidth - (margin * 3)) / 2; // Half page width minus margins
      const maxHeight = 70; // Limit height
      
      let imgWidth = maxWidth;
      let imgHeight = (originalImg.height * maxWidth) / originalImg.width;
      
      if (imgHeight > maxHeight) {
        imgHeight = maxHeight;
        imgWidth = (originalImg.width * maxHeight) / originalImg.height;
      }
      
      // Add the original image to the first page
      doc.addImage(originalImg, 'JPEG', margin, yPosition, imgWidth, imgHeight);
      
      // Add label under the original image
      doc.setFontSize(8);
      doc.setFont("helvetica", "italic");
      doc.text("Original X-ray", margin + imgWidth/2, yPosition + imgHeight + 5, { align: "center" });
      
      // Start a new page for all diseases
      doc.addPage();
      let currentYPosition = 30; // Start at the top of the new page
      
      // Add Disease Findings title on the new page
      doc.setFontSize(16);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(41, 98, 255);
      doc.text("Disease Findings", margin, currentYPosition);
      
      // Add a separator line
      doc.setDrawColor(220, 220, 220);
      doc.setLineWidth(0.3);
      doc.line(margin, currentYPosition + 4, pageWidth - margin, currentYPosition + 4);
      
      currentYPosition += 15; // Space after title
      
      // Get all predictions with bounding boxes
      const predictionsWithBbox = Array.isArray(result.predictions) 
        ? result.predictions.filter((p: any) => p.hasBbox || p.bbox)
        : [];
      
      // Create a canvas element for the heatmap
      const canvas = document.createElement('canvas');
      canvas.width = originalImg.width;
      canvas.height = originalImg.height;
      
      // Process each disease
      for (let i = 0; i < predictionsWithBbox.length; i++) {
        const prediction = predictionsWithBbox[i];
        const probability = Math.round((prediction.probability || 0) * 100);
        
        // Check if we need to add a new page for this disease
        // If the remaining space on the page is not enough for disease title + images + some margin
        if (currentYPosition + imgHeight + 40 > pageHeight - margin) {
          // Add a new page
          doc.addPage();
          currentYPosition = 30; // Start position on new page
        }
        
        // Add disease title
        doc.setFontSize(14);
        doc.setFont("helvetica", "bold");
        doc.setTextColor(41, 98, 255);
        doc.text(`${prediction.disease} (${probability}%)`, margin, currentYPosition);
        
        // Add a separator line under the disease title
        doc.setDrawColor(220, 220, 220);
        doc.setLineWidth(0.3);
        doc.line(margin, currentYPosition + 4, pageWidth - margin, currentYPosition + 4);
        
        currentYPosition += 15; // Space after title
        
        // Draw the heatmap
        const ctx = canvas.getContext('2d');
        if (ctx) {
          // Clear canvas
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          
          // Draw the original image on the canvas
          ctx.drawImage(originalImg, 0, 0);
          
          // Get bbox directly from model output
          const bbox = prediction.bbox || prediction;
          
          // Use coordinates directly
          if (!bbox) {
            console.warn(`No bounding box available for ${prediction.disease}`);
            continue;
          }
          
          // Determine color based on disease type
          let baseColor = "255, 0, 0"; // Default red
          if (prediction.disease === "Pneumonia")
            baseColor = "255, 0, 0"; // Red
          else if (prediction.disease === "Effusion")
            baseColor = "0, 0, 255"; // Blue
          else if (prediction.disease === "Cardiomegaly")
            baseColor = "255, 165, 0"; // Orange
          else if (prediction.disease === "Atelectasis")
            baseColor = "128, 0, 128"; // Purple
          else if (prediction.disease === "Mass")
            baseColor = "0, 128, 0"; // Green
          else if (prediction.disease === "Nodule")
            baseColor = "255, 255, 0"; // Yellow
          else if (prediction.disease === "Pneumothorax")
            baseColor = "0, 255, 255"; // Cyan
          else if (prediction.disease === "Infiltration") 
            baseColor = "255, 0, 255"; // Magenta
          
          // Generate heatmap overlay for this specific disease
          createHeatmap(
            ctx, 
            bbox.x, 
            bbox.y, 
            bbox.width, 
            bbox.height, 
            originalImg.width, 
            originalImg.height,
            baseColor
          );
          
          // Convert canvas to data URL
          const heatmapDataUrl = canvas.toDataURL('image/jpeg');
          
          // Add the heatmap image in the center
          const centeredX = (pageWidth - imgWidth) / 2;
          doc.addImage(heatmapDataUrl, 'JPEG', centeredX, currentYPosition, imgWidth, imgHeight);
          
          // Add label under the heatmap image
          doc.setFontSize(8);
          doc.setFont("helvetica", "italic");
          doc.text(`Heatmap: ${prediction.disease}`, centeredX + imgWidth/2, currentYPosition + imgHeight + 5, { align: "center" });
          
          // Add findings below the images
          doc.setFontSize(10);
          doc.setFont("helvetica", "normal");
          const findingsText = `Findings: ${prediction.disease} detected with ${probability}% probability`;
          doc.text(findingsText, pageWidth / 2, currentYPosition + imgHeight + 15, { align: "center" });
          
          // Update currentYPosition to after this disease, with minimal spacing
          currentYPosition += imgHeight + 25;
          
          // Add a small separator between diseases (except for the last one)
          if (i < predictionsWithBbox.length - 1) {
            doc.setDrawColor(220, 220, 220);
            doc.setLineWidth(0.3);
            doc.line(margin, currentYPosition - 5, pageWidth - margin, currentYPosition - 5);
            currentYPosition += 10; // Small gap after separator
          }
        }
      }
      
      // Update the main yPosition to continue with the rest of the document
      yPosition = 40; // Reset to continue with Analysis Results on a new page
      doc.addPage(); // Add a new page for Analysis Results and remaining sections
      
      // Analysis Results Section
      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(41, 98, 255);
      doc.text("Analysis Results", margin, yPosition);
      
      yPosition += 8;
      
      // Add AI analysis results table
      const tableColumn = ["Finding", "Probability", "Severity"];
      
      // Check if predictions is an array and handle accordingly
      let tableRows = [];
      if (result && result.predictions) {
        if (Array.isArray(result.predictions)) {
          // Sort predictions by probability (highest first)
          const sortedPredictions = [...result.predictions].sort((a, b) => b.probability - a.probability);
          
          tableRows = sortedPredictions.map((prediction: any) => {
            const probability = Math.round((prediction.probability || 0) * 100);
            let severity = "Low";
            
            if (probability > 70) severity = "High";
            else if (probability > 50) severity = "Moderate";
            
            return [
              prediction.disease || "Unknown",
              `${probability}%`,
              severity
            ];
          });
        }
      }
      
      if (tableRows.length === 0) {
        tableRows = [["No findings detected", "0%", "None"]];
      }
      
      autoTable(doc, {
        head: [tableColumn],
        body: tableRows,
        startY: yPosition,
        theme: "grid",
        styles: { 
          fontSize: 10,
          cellPadding: 3,
          halign: "center" 
        },
        headStyles: { 
          fillColor: [41, 98, 255], 
          textColor: [255, 255, 255],
          fontStyle: 'bold'
        },
        columnStyles: {
          0: { halign: 'left' }
        },
        alternateRowStyles: {
          fillColor: [245, 247, 250]
        },
        margin: { left: margin, right: margin }
      });
      
      yPosition = (doc as any).lastAutoTable.finalY + 10;
      
      // Clinical Impression / Doctor's Notes
      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(41, 98, 255);
      doc.text("Clinical Impression", margin, yPosition);
      
      yPosition += 6;
      
      // Add a box for the doctor's notes
      doc.setDrawColor(220, 220, 220);
      doc.setFillColor(250, 250, 250);
      doc.roundedRect(margin, yPosition, pageWidth - (margin * 2), 40, 2, 2, 'FD');
      
      doc.setFontSize(10);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(60, 60, 60);
      
      if (result.doctor_notes) {
        // Split text to fit within the box with word wrapping
        const splitText = doc.splitTextToSize(result.doctor_notes, pageWidth - (margin * 2) - 10);
        doc.text(splitText, margin + 5, yPosition + 5);
      } else {
        doc.setFontSize(10);
        doc.setFont("helvetica", "italic");
        doc.setTextColor(100, 100, 100);
        doc.text("No clinical notes provided.", margin + 5, yPosition + 5);
      }
      
      yPosition += 50;
      
      // Recommendations
      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(41, 98, 255);
      doc.text("Recommendations", margin, yPosition);
      
      yPosition += 6;
      
      // Generate some recommendations based on findings
      let recommendations = ["Follow up with primary care physician."];
      
      if (tableRows.length > 0 && tableRows[0][0] !== "No findings detected") {
        // Add specific recommendations based on findings
        const highSeverityFindings = tableRows.filter(row => row[2] === "High");
        if (highSeverityFindings.length > 0) {
          recommendations = [
            "Urgent follow-up with specialist recommended.",
            "Consider additional diagnostic testing.",
            "Monitor patient closely for symptom progression."
          ];
        } else {
          recommendations = [
            "Routine follow-up in 3-6 months recommended.",
            "Monitor for any changes in symptoms.",
            "Consider lifestyle modifications as appropriate."
          ];
        }
      }
      
      // Add recommendations as bullet points
      doc.setFontSize(10);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(60, 60, 60);
      
      recommendations.forEach((rec, index) => {
        doc.text(`• ${rec}`, margin + 5, yPosition + (index * 6));
      });
      
      yPosition += 6 * recommendations.length + 15;
      
      // Add signature line
      doc.setDrawColor(100, 100, 100);
      doc.setLineWidth(0.5);
      doc.line(margin, yPosition, margin + 60, yPosition);
      
      doc.setFontSize(10);
      doc.text("Physician's Signature", margin, yPosition + 5);
      
      // Add date line
      doc.line(pageWidth - margin - 60, yPosition, pageWidth - margin, yPosition);
      doc.text("Date", pageWidth - margin - 20, yPosition + 5);

      // Add footer
      doc.setFontSize(8);
      doc.setTextColor(150, 150, 150);
        doc.text(
        "This report was generated with AI assistance and should be reviewed by a qualified healthcare professional.",
        pageWidth / 2,
        pageHeight - 15,
        { align: "center" }
      );
      
      doc.text(
        `Generated by DiagnoLink System - ${new Date().toLocaleString()}`,
        pageWidth / 2,
        pageHeight - 10,
        { align: "center" }
      );

      // Save the PDF
      doc.save(`Medical_Report_${patientData.name}_${date.replace(/\//g, '-')}.pdf`);

      toast({
        title: "Report generated successfully",
        description: "The PDF report has been downloaded",
      });
    } catch (error) {
      console.error("Error generating PDF:", error);
      console.error("Error stack:", error instanceof Error ? error.stack : "No stack trace");
      toast({
        title: "Failed to generate report",
        description: "An error occurred while generating the report. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <Button onClick={generatePdf} disabled={isGenerating} variant="outline">
      {isGenerating ? (
        <>
          <LucideLoader className="mr-2 h-4 w-4 animate-spin" />
          Generating report...
        </>
      ) : (
        <>
          <LucideFileText className="mr-2 h-4 w-4" />
          Download PDF Report
        </>
      )}
    </Button>
  );
}
